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

    public addMessage(role: string, text: string) {
        if (this._view) {
            this._view.webview.postMessage({ type: 'addMessage', role, text });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', 'markdown-it', 'dist', 'markdown-it.min.js'));

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Achilles Chat</title>
                <style>
                    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 10px; }
                    #chat { margin-bottom: 10px; height: calc(100vh - 80px); overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
                    .message { padding: 8px; border-radius: 4px; max-width: 90%; word-wrap: break-word; }
                    .user { background: var(--vscode-button-background); color: var(--vscode-button-foreground); align-self: flex-end; }
                    .bot { background: var(--vscode-sideBar-background); border: 1px solid var(--vscode-panel-border); align-self: flex-start; }
                    code { background: rgba(128, 128, 128, 0.2); padding: 2px 4px; border-radius: 3px; }
                    pre { background: rgba(0, 0, 0, 0.3); padding: 8px; overflow-x: auto; border-radius: 4px; }
                    input { width: 100%; box-sizing: border-box; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); padding: 8px; }
                </style>
            </head>
            <body>
                <div id="chat"></div>
                <input type="text" id="input" placeholder="Ask Achilles..." />
                <script src="${scriptUri}"></script>
                <script>
                    const vscode = acquireVsCodeApi();
                    const chat = document.getElementById('chat');
                    const input = document.getElementById('input');

                    window.addEventListener('message', event => {
                        const message = event.data;
                        if (message.type === 'addMessage') {
                            const div = document.createElement('div');
                            div.className = 'message ' + (message.role === 'user' ? 'user' : 'bot');
                            div.textContent = message.text; // Default to text

                            // Use markdown-it if available (loaded via CDN for simplicity in this webview)
                            if (window.markdownit) {
                                const md = window.markdownit();
                                div.innerHTML = md.render(message.text);
                            }

                            chat.appendChild(div);
                            chat.scrollTop = chat.scrollHeight;
                        }
                    });

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
