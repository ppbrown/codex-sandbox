
# Run with   python -m pytest tests

import pytest

from dbsim import DBSim


class TestDBSim:
    def test_set_requires_transaction(self):
        db = DBSim()
        with pytest.raises(RuntimeError):
            db.set("key", "value")

    def test_commit_requires_transaction(self):
        db = DBSim()
        with pytest.raises(RuntimeError):
            db.commit()

    def test_rollback_requires_transaction(self):
        db = DBSim()
        with pytest.raises(RuntimeError):
            db.rollback()

    def test_single_transaction_set_and_commit(self):
        db = DBSim()
        db.begin_transaction()
        db.set("name", "alice")
        assert db.get("name") == "alice"

        db.commit()
        assert db.get("name") == "alice"

        # after commit, set should fail until a new transaction is opened
        with pytest.raises(RuntimeError):
            db.set("name", "bob")

    def test_nested_transactions_commit_all(self):
        db = DBSim()
        db.begin_transaction()
        db.set("name", "outer")

        db.begin_transaction()
        db.set("name", "inner")
        db.set("age", 5)
        assert db.get("name") == "inner"
        assert db.get("age") == 5

        # committing should apply both layers and close all transactions
        db.commit()
        assert db.get("name") == "inner"
        assert db.get("age") == 5

        with pytest.raises(RuntimeError):
            db.rollback()

    def test_nested_transactions_rollback_only_latest(self):
        db = DBSim()
        db.begin_transaction()
        db.set("counter", 1)

        db.begin_transaction()
        db.set("counter", 2)
        db.rollback()

        # outer transaction value should still be present
        assert db.get("counter") == 1

        # committing should persist the outer value
        db.commit()
        assert db.get("counter") == 1

    def test_base_store_visible_in_transactions(self):
        db = DBSim()
        db.begin_transaction()
        db.set("flag", True)
        db.commit()

        db.begin_transaction()
        assert db.get("flag") is True
        db.set("flag", False)
        assert db.get("flag") is False
        db.rollback()

        assert db.get("flag") is True
