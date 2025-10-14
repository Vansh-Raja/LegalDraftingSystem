"""
Database debug utilities for the Legal Drafting System.
Provides command-line tools to manage and inspect the PGVector database.
"""

from rag import COLLECTION_NAME, _get_pg_connection_string
import psycopg2


def menu():
    """Display the debug utilities menu."""
    print("Debug Utilities:")
    print("  1) Clear PGVector collection (dangerous - deletes all embeddings)")
    print("  2) Count documents in collection")
    print("  3) Count distinct cases (by file_stem)")
    print("  4) List case file stems (compact ranges)")
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
        elif c == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or q.")
        print()


