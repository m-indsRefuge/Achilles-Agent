import * as vscode from 'vscode';

export class SidebarProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'achilles.sidebar';
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(data => {
            switch (data.type) {
                case 'onAsk': {
                    vscode.commands.executeCommand('achilles.ask', data.value);
                    break;
                }
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Achilles Chat</title>
                <style>
                    body { font-family: sans-serif; padding: 10px; }
                    #chat { margin-bottom: 10px; border: 1px solid #ccc; height: 200px; overflow-y: auto; padding: 5px; }
                    input { width: 100%; box-sizing: border-box; }
                </style>
            </head>
            <body>
                <div id="chat">Welcome to Achilles Agent!</div>
                <input type="text" id="input" placeholder="Ask Achilles..." />
                <script>
                    const vscode = acquireVsCodeApi();
                    const input = document.getElementById('input');
                    input.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') {
                            vscode.postMessage({ type: 'onAsk', value: input.value });
                            input.value = '';
                        }
                    });
                </script>
            </body>
            </html>`;
    }
}
