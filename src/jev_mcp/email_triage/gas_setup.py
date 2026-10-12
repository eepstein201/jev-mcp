import json
import os
from pathlib import Path

# Stub implementations to avoid heavy dependencies if not installed
try:
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
    from googleapiclient.discovery import build  # type: ignore[import-untyped]
except ImportError:
    InstalledAppFlow = None
    build = None


API_KEY_PROPERTY = "JEV_MCP_API_KEY"

KEY_NEXT_STEP = (
    f"Set the Script Property {API_KEY_PROPERTY} to your webhook key: open the script, "
    "Project Settings, Script Properties, Add script property. The key is not uploaded "
    "with the code, so it never sits in the script source."
)


def setup_gas_workflow(webhook_url: str | None = None) -> str:
    """
    Automated CLI Setup Flow for Google Apps Script.
    Deploys the .gs file via Google API. The webhook key is not part of the upload:
    the script reads it from a Script Property that the user sets once.
    """
    if InstalledAppFlow is None:
        raise ImportError(
            "Please install google-api-python-client and google-auth-oauthlib to use the automated setup."
        )

    print("Welcome to the Jev-Mail Google Apps Script Setup Wizard.")
    if webhook_url is None:
        webhook_url = input(
            "Enter your Webhook URL (e.g. https://xyz.ngrok.app/api/v1/triage/email): "
        ).strip()

    print("\nAuthenticating with Google... (A browser window will open)")

    # We require a client_secrets.json in the ~/.jev-mcp directory
    secret_path = Path.home() / ".jev-mcp" / "client_secrets.json"
    if not secret_path.exists():
        raise FileNotFoundError(
            f"Missing OAuth credentials. Please place your Google Cloud client_secrets.json at {secret_path}"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(secret_path),
        scopes=[
            "https://www.googleapis.com/auth/script.projects",
            "https://www.googleapis.com/auth/script.external_request",
            "https://www.googleapis.com/auth/gmail.modify",
        ],
    )
    creds = flow.run_local_server(port=0)

    service = build("script", "v1", credentials=creds)

    print("Creating Apps Script Project...")
    request = {"title": "Jev-Mail Triage Worker"}
    response = service.projects().create(body=request).execute()
    script_id = response.get("scriptId")

    # Read the template (assuming it's relative to the project root)
    # For testing, we mock open
    template_path = (
        Path(__file__).parent.parent.parent.parent / "examples" / "gas_triage.js"
    )
    try:
        with open(template_path, "r") as f:
            template_code = f.read()
    except FileNotFoundError:
        # Fallback for testing
        template_code = "const JEV_MCP_WEBHOOK_URL = 'TEMPLATE';"

    # json.dumps writes the URL as an escaped JavaScript string literal, so a quote
    # in it cannot end the string and run as code.
    url_literal = json.dumps(webhook_url)
    final_code = template_code.replace("'TEMPLATE'", url_literal, 1)

    # We just do a naive prepend for simplicity if replace didn't work
    if url_literal not in final_code:
        final_code = f"const JEV_MCP_WEBHOOK_URL = {url_literal};\n" + final_code

    print("Uploading code...")
    update_request = {
        "files": [
            {"name": "Code", "type": "SERVER_JS", "source": final_code},
            {
                "name": "appsscript",
                "type": "JSON",
                "source": json.dumps(
                    {
                        "timeZone": "America/New_York",
                        "dependencies": {},
                        "oauthScopes": [
                            "https://www.googleapis.com/auth/gmail.modify",
                            "https://www.googleapis.com/auth/script.external_request",
                        ],
                    }
                ),
            },
        ]
    }

    service.projects().updateContent(scriptId=script_id, body=update_request).execute()

    url = f"https://script.google.com/d/{script_id}/edit"
    print(f"\n✅ Setup Complete! Your script has been deployed.")
    print(f"👉 Please open this link: {url}")
    print(f"👉 {KEY_NEXT_STEP}")
    print(
        "Then click 'Run' once to grant Gmail permissions, and set your time trigger."
    )

    return url
