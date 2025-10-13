from rag import COLLECTION_NAME, _get_pg_connection_string
import psycopg2


def menu():
    print("Debug Utilities:\n"
          "1) Clear PGVector collection (dangerous)\n"
          "2) Count documents in collection\n"
          "q) Quit\n")


def _exec_sql(sql: str, params: tuple | None = None):
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
    try:
        # Delete embeddings for the named collection, then remove the collection row
        _exec_sql(
            "DELETE FROM langchain_pg_embedding WHERE collection_id IN (SELECT uuid FROM langchain_pg_collection WHERE name=%s)",
            (COLLECTION_NAME,),
        )
        _exec_sql("DELETE FROM langchain_pg_collection WHERE name=%s", (COLLECTION_NAME,))
        print(f"Collection '{COLLECTION_NAME}' cleared.")
    except Exception as e:
        print(f"Error clearing collection: {e}")


def count_documents():
    try:
        rows = _exec_sql(
            "SELECT COUNT(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id = c.uuid WHERE c.name=%s",
            (COLLECTION_NAME,),
        )
        cnt = rows[0][0] if rows else 0
        print(f"Collection '{COLLECTION_NAME}' embeddings: {cnt}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    while True:
        menu()
        c = input("Select option > ").strip().lower()
        if c == "1":
            clear_collection()
        elif c == "2":
            count_documents()
        elif c == "q":
            break
        else:
            print("Invalid choice.")


