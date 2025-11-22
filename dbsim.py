class DBSim:
    """Simulates a database with basic transactional semantics."""

    def __init__(self):
        self._store = {}
        self._transactions = []

    def begin_transaction(self):
        """Start a new transaction layer."""
        self._transactions.append({})

    def _require_transaction(self):
        if not self._transactions:
            raise RuntimeError("No active transaction")

    def get(self, key, default=None):
        """Retrieve the value for *key* from the newest open transaction or base store."""
        # Search from the most recent transaction backwards
        for layer in reversed(self._transactions):
            if key in layer:
                return layer[key]
        return self._store.get(key, default)

    def set(self, key, value):
        """Set *key* to *value* in the current transaction."""
        self._require_transaction()
        self._transactions[-1][key] = value

    def commit(self):
        """Commit all open transactions into the base store, oldest first."""
        self._require_transaction()
        for layer in self._transactions:
            self._store.update(layer)
        self._transactions.clear()

    def rollback(self):
        """Rollback only the most recent transaction."""
        self._require_transaction()
        self._transactions.pop()
