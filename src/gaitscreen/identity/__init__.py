"""Who is walking: suggestions from faces and walk signatures, for staff to confirm.

Shared by the app and the Raspberry Pi device so both describe a walk the
same way. The list of enrolled people, and the matching against it, live only
on the platform; the device computes a walk's signature and faces and sends
them with its result.
"""
from .face import PhotoError, embed_photo, face_embeddings, load_models
from .matcher import Candidate, PersonGallery, rank, unit
from .signature import signature_from_result, walk_signature

__all__ = ["Candidate", "PersonGallery", "PhotoError", "embed_photo", "face_embeddings",
           "load_models", "rank", "signature_from_result", "unit", "walk_signature"]
