import * as vscode from 'vscode';

export class EditorTools {
    /**
     * Retrieves all diagnostics (errors, warnings) for a given file.
     */
    public static async getDiagnostics(uri: vscode.Uri): Promise<string> {
        const diagnostics = vscode.languages.getDiagnostics(uri);
        return diagnostics
            .map(d => `[${vscode.DiagnosticSeverity[d.severity]}] Line ${d.range.start.line}: ${d.message}`)
            .join('\n');
    }

    /**
     * Retrieves workspace-wide symbols matching a query.
     */
    public static async getSymbols(query: string): Promise<string> {
        const symbols = await vscode.commands.executeCommand<vscode.SymbolInformation[]>(
            'vscode.executeWorkspaceSymbolProvider',
            query
        );
        if (!symbols) return 'No symbols found.';
        return symbols
            .slice(0, 20)
            .map(s => `${s.name} (${vscode.SymbolKind[s.kind]}) at ${s.location.uri.fsPath}`)
            .join('\n');
    }
}
