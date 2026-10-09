from app.services.vector_storage import configure_test_mode

# Configure isolated in-memory Qdrant before tests access vector storage.
configure_test_mode(True)