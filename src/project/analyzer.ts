import * as vscode from 'vscode';
import * as fs from 'fs';
import { PythonBridge } from '../comms/pythonBridge';

export class ProjectAnalyzer {
    private bridge: PythonBridge;

    constructor() {
        this.bridge = new PythonBridge();
    }

    public async analyzeWorkspace(): Promise<void> {
        // Use vscode.workspace.findFiles for better performance and respect for .gitignore
        const files = await vscode.workspace.findFiles(
            '**/*.{ts,js,py,txt,md}',
            '**/node_modules/**'
        );

        for (const file of files) {
            await this.indexFile(file.fsPath);
        }
    }

    private async indexFile(filePath: string): Promise<void> {
        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const lines = content.split('\n');

            const chunks: string[] = [];
            const metadatas: any[] = [];

            // Chunk and prepare for batch indexing
            for (let i = 0; i < lines.length; i += 50) {
                const chunk = lines.slice(i, i + 50).join('\n');
                if (chunk.trim()) {
                    chunks.push(chunk);
                    metadatas.push({ path: filePath, lineStart: i });
                }
            }

            if (chunks.length > 0) {
                await this.bridge.queryMemory('kb_add_batch', {
                    texts: chunks,
                    metadatas: metadatas
                });
            }
        } catch (error) {
            console.error(`Error indexing file ${filePath}:`, error);
        }
    }
}
