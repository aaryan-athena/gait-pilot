"""Streamlit pilot platform for gaitscreen.

Everything the web UI needs lives in this folder:

    app/
        main.py       entry point -- `streamlit run app/main.py`
        shared.py     config, database access and widgets used by several views
        views/        one module per page

The UI is a thin layer. It calls the same ``gaitscreen`` pipeline as the CLI and
adds no analysis of its own, so a result seen in the app and a result from
``gaitscreen analyze`` are the same numbers.

The directory is deliberately named ``views`` rather than ``pages``: Streamlit
treats a ``pages/`` folder beside the entry script as an automatic multi-page
app and builds its own navigation, which would sit alongside the sidebar
navigation this app already provides.
"""
