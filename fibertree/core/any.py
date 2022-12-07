#cython: language_level=3
"""Any

A class whose instances are always equal to other objects

"""

class Any:
    """Class that is always equal to other objects """
    def __eq__(self, other):
        return True

    def __repr__(self):
        return "Any"

# Only ever instantiate one Any
ANY = Any()
