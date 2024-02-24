### Packages and Modules
- Make sure to always add a __init__.py file in every directory so that python knows its a pacakge
- It will help you add imports from other directories e.g. parent, sibling directory, importing from them otherwise would result an error
- You can easily run the file at top most level, but to run the scripts inside packages and subpackages (called modules) use this command `python3 -m mypackage.mymodule`