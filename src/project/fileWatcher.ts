import * as vscode from 'vscode';
import { ProjectAnalyzer } from './analyzer';

export class FileWatcher {
    private watcher: vscode.FileSystemWatcher;
    private analyzer: ProjectAnalyzer;

    constructor(analyzer: ProjectAnalyzer) {
        this.analyzer = analyzer;
        // Watch all source files in the workspace
        this.watcher = vscode.workspace.createFileSystemWatcher('**/*.{ts,js,py,txt,md}');

        this.watcher.onDidChange(async (uri) => {
            console.log(`File changed: ${uri.fsPath}`);
            await this.analyzer.indexFile(uri.fsPath);
        });

        this.watcher.onDidCreate(async (uri) => {
            console.log(`File created: ${uri.fsPath}`);
            await this.analyzer.indexFile(uri.fsPath);
        });
    }

    public dispose() {
        this.watcher.dispose();
    }
}
