from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self
import pickle
import dill._dill as dill_dill
import datasets.utils._dill as datasets_dill


# ⚠️ Ces deux chaînes font partie intégrante du PROMPT sur lequel le modèle a été entraîné (en anglais).
# Il ne faut donc PAS les traduire, sous peine de dégrader fortement les prédictions.
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


def _ensure_pickle_compat_for_datasets() -> None:
    """Patch de compatibilité Python 3.14 pour dill/datasets (appel legacy à _batch_setitems)."""
    patch_version = 2
    if globals().get("_PICKLE_COMPAT_PATCHED_VERSION") == patch_version:
        return

    def _patch_pickler_cls(pickler_cls) -> None:
        batch_setitems = getattr(pickler_cls, "_batch_setitems", None)
        if batch_setitems is None:
            return

        try:
            argcount = batch_setitems.__code__.co_argcount
        except AttributeError:
            return

        # Signature moderne : (self, items). On ajoute une compat pour ignorer l'argument legacy "obj".
        if argcount == 2:
            original_batch_setitems = batch_setitems

            def _batch_setitems_compat(self, items, obj=None):
                return original_batch_setitems(self, items)

            pickler_cls._batch_setitems = _batch_setitems_compat

    _patch_pickler_cls(pickle._Pickler)
    _patch_pickler_cls(dill_dill.StockPickler)
    _patch_pickler_cls(dill_dill.Pickler)

    # En Python 3.14, datasets.utils._dill.Pickler._batch_setitems a une
    # signature incompatible avec pickle._Pickler.save_dict.
    # On remplace donc cette surcharge par l'implémentation standard.
    datasets_dill.Pickler._batch_setitems = pickle._Pickler._batch_setitems

    globals()["_PICKLE_COMPAT_PATCHED_VERSION"] = patch_version


class Item(BaseModel):
    """
    Un Item représente un point de données : un Produit associé à un Prix.
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        # Construit un prompt COMPLET (question + texte + prix réel) pour l'entraînement
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        # Construit un prompt de TEST : tout jusqu'au préfixe "Price is $", sans le prix (le modèle doit le compléter)
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        # Représentation lisible, ex. : <Excess V2 Pedal = $219.0>
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Publie les listes d'Items (train/validation/test) sur le Hub Hugging Face."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Charge un jeu de données depuis le Hub Hugging Face et reconstruit les objets Item."""
        _ensure_pickle_compat_for_datasets()
        try:
            ds = load_dataset(dataset_name)
        except TypeError as exc:
            # Contournement pour certaines combinaisons Python 3.14 / datasets / dill
            # qui échouent pendant la phase de sérialisation interne.
            if "Pickler._batch_setitems" not in str(exc):
                raise
            ds = load_dataset(dataset_name, streaming=True)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
