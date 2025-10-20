"""
Database debug utilities for the Legal Drafting System.
Provides command-line tools to manage and inspect the PGVector database.
"""

from collections import defaultdict
import json
from pathlib import Path

from rag import COLLECTION_NAME, _get_pg_connection_string
import psycopg2


def menu():
    """Display the debug utilities menu."""
    print("Debug Utilities:")
    print("  1) Clear PGVector collection (dangerous - deletes all embeddings)")
    print("  2) Count documents in collection")
    print("  3) Count distinct cases (by file_stem)")
    print("  4) List case file stems (compact ranges)")
    print("  5) Check duplicate metadata (files + database)")
    print("  q) Quit")
    print()


def _exec_sql(sql: str, params: tuple | None = None):
    """
    Execute SQL query against the PostgreSQL database.
    
    Args:
        sql (str): SQL query to execute
        params (tuple | None): Query parameters
        
    Returns:
        List of rows or None if no results
    """
    conn_str = _get_pg_connection_string()
    conn = psycopg2.connect(conn_str)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            try:
                rows = cur.fetchall()
            except Exception:
                rows = None
            return rows
    finally:
        conn.close()


def clear_collection():
    """
    Clear all embeddings from the PGVector collection.
    WARNING: This permanently deletes all vector embeddings!
    """
    try:
        # Delete all embeddings for the collection
        _exec_sql(
            "DELETE FROM langchain_pg_embedding WHERE collection_id IN (SELECT uuid FROM langchain_pg_collection WHERE name=%s)",
            (COLLECTION_NAME,),
        )
        # Delete the collection record itself
        _exec_sql("DELETE FROM langchain_pg_collection WHERE name=%s", (COLLECTION_NAME,))
        print(f"Collection '{COLLECTION_NAME}' cleared successfully.")
    except Exception as e:
        print(f"Error clearing collection: {e}")


def count_documents():
    """Count the number of document embeddings in the collection."""
    try:
        rows = _exec_sql(
            "SELECT COUNT(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id = c.uuid WHERE c.name=%s",
            (COLLECTION_NAME,),
        )
        cnt = rows[0][0] if rows else 0
        print(f"Collection '{COLLECTION_NAME}' contains {cnt} document embeddings.")
    except Exception as e:
        print(f"Error counting documents: {e}")


def _get_distinct_file_stems():
    """Return a set of distinct file_stem values stored in cmetadata for the collection."""
    try:
        rows = _exec_sql(
            "SELECT DISTINCT e.cmetadata->>'file_stem' AS stem "
            "FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name=%s AND e.cmetadata ? 'file_stem'",
            (COLLECTION_NAME,),
        )
        stems = set()
        if rows:
            for (stem,) in rows:
                if stem is not None and stem != "":
                    stems.add(stem)
        return stems
    except Exception as e:
        print(f"Error fetching stems: {e}")
        return set()


def count_distinct_cases():
    """Count unique cases (distinct file_stem) in the collection."""
    stems = _get_distinct_file_stems()
    print(f"Collection '{COLLECTION_NAME}' distinct cases: {len(stems)}")


def _compact_ranges(nums: list[int]) -> str:
    """Return a compact range string like '1-5, 7, 10-12' for a sorted list of ints."""
    if not nums:
        return ""
    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        # close range
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")
        start = prev = n
    # final range
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
    return ", ".join(ranges)


def list_case_file_stems():
    """List case file stems; show numeric stems as compact ranges; non-numeric separately."""
    stems = _get_distinct_file_stems()
    if not stems:
        print("No stems found.")
        return
    numeric = []
    other = []
    for s in stems:
        if s.isdigit():
            try:
                numeric.append(int(s))
            except Exception:
                other.append(s)
        else:
            other.append(s)
    numeric.sort()
    other_sorted = sorted(other)
    compact = _compact_ranges(numeric)
    print(f"Distinct cases total: {len(stems)}")
    if numeric:
        print(f"Numeric stems ({len(numeric)}): {compact}")
    if other_sorted:
        print(f"Non-numeric stems ({len(other_sorted)}): {', '.join(other_sorted)}")


def _scan_local_metadata(root: Path) -> dict[str, list[str]]:
    """
    Return mapping from case_number to stems for metadata JSON files under root.
    """
    duplicates: dict[str, list[str]] = defaultdict(list)
    all_items: dict[str, list[str]] = defaultdict(list)
    for json_path in root.glob("*.json"):
        stem = json_path.stem
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_number = (payload.get("case_number") or "").strip()
        if case_number:
            all_items[case_number].append(stem)
    for case, stems in all_items.items():
        unique_stems = sorted(set(stems))
        if len(unique_stems) > 1:
            duplicates[case] = unique_stems
    return duplicates


def _scan_db_duplicates() -> dict[str, list[str]]:
    """
    Return mapping of duplicated case_number entries stored in PGVector.
    """
    duplicates: dict[str, list[str]] = {}
    try:
        rows = _exec_sql(
            "SELECT e.cmetadata->>'case_number' AS case_number, array_agg(DISTINCT e.cmetadata->>'file_stem') "
            "FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name=%s AND e.cmetadata ? 'case_number' "
            "GROUP BY case_number HAVING COUNT(DISTINCT e.cmetadata->>'file_stem') > 1",
            (COLLECTION_NAME,),
        )
    except Exception as exc:
        print(f"Error scanning database duplicates: {exc}")
        return duplicates

    if rows:
        for case_number, stems in rows:
            key = (case_number or "").strip()
            if key:
                ordered = sorted({stem for stem in (stems or []) if stem})
                if ordered:
                    duplicates[key] = ordered
    return duplicates


def check_duplicate_metadata():
    """
    Check for duplicate case_number metadata locally and in the PGVector collection.
    """
    metadata_dir = Path("processed_data/metadata")
    if not metadata_dir.exists():
        print("No local metadata directory found at processed_data/metadata.")
    else:
        local_dupes = _scan_local_metadata(metadata_dir)
        if not local_dupes:
            print("Local metadata files: no duplicate case_number entries detected.")
        else:
            print(f"Local metadata files: {len(local_dupes)} duplicated case_number entries detected.")
            for case, stems in sorted(local_dupes.items()):
                snippet = ", ".join(stems[:10])
                print(f"  - {case}: stems {snippet}")
                if len(stems) > 10:
                    print(f"    (and {len(stems) - 10} more)")

    db_dupes = _scan_db_duplicates()
    if not db_dupes:
        print("Database metadata: no duplicate case_number entries detected.")
    else:
        print(f"Database metadata: {len(db_dupes)} duplicated case_number entries detected.")
        for case, stems in sorted(db_dupes.items()):
            snippet = ", ".join(stems[:10])
            print(f"  - {case}: stems {snippet}")
            if len(stems) > 10:
                print(f"    (and {len(stems) - 10} more)")


if __name__ == "__main__":
    print("Legal Drafting System - Database Debug Utilities")
    print("=" * 50)
    
    while True:
        menu()
        c = input("Select option > ").strip().lower()
        if c == "1":
            confirm = input("Are you sure you want to clear the collection? (yes/no): ").strip().lower()
            if confirm == "yes":
                clear_collection()
            else:
                print("Operation cancelled.")
        elif c == "2":
            count_documents()
        elif c == "3":
            count_distinct_cases()
        elif c == "4":
            list_case_file_stems()
        elif c == "5":
            check_duplicate_metadata()
        elif c == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, 5, or q.")
        print()
