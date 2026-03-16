import * as vscode from 'vscode';
import * as fs from 'fs';
import { PythonBridge } from '../comms/pythonBridge';

export class ProjectAnalyzer {
    private bridge: PythonBridge;

    constructor(bridge: PythonBridge) {
        this.bridge = bridge;
    }

    public async analyzeWorkspace(): Promise<void> {
        const files = await vscode.workspace.findFiles(
            '**/*.{ts,js,py,txt,md}',
            '**/node_modules/**'
        );

        for (const file of files) {
            await this.indexFile(file.fsPath);
        }
    }

    public async indexFile(filePath: string): Promise<void> {
        try {
            // Clear existing entries for this file to prevent duplicates
            await this.bridge.queryMemory('kb_clear_file', { path: filePath });

            const content = fs.readFileSync(filePath, 'utf-8');
            const lines = content.split('\n');

            const chunks: string[] = [];
            const metadatas: any[] = [];

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
