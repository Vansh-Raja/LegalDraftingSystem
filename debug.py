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
        elif c == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or q.")
        print()


