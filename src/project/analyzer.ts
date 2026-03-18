import * as vscode from 'vscode';
import * as fs from 'fs';
import * as crypto from 'crypto';
import { PythonBridge } from '../comms/pythonBridge';
import { ProgressManager } from '../ui/notifications';

export class ProjectAnalyzer {
    private bridge: PythonBridge;

    constructor(bridge: PythonBridge) {
        this.bridge = bridge;
    }

    public async analyzeWorkspace(): Promise<void> {
        await ProgressManager.withProgress('Achilles: Analyzing Project', async (progress) => {
            const files = await vscode.workspace.findFiles(
                '**/*.{ts,js,py,txt,md}',
                '**/node_modules/**'
            );

            // Fetch existing file hashes from backend (if available) for incremental indexing
            const indexedDocs: any = await this.bridge.queryMemory('kb', { text: '', top_k: 1000 });
            const fileHashes = new Map<string, string>();
            indexedDocs.forEach((doc: any) => {
                if (doc.metadata && doc.metadata.path && doc.metadata.file_hash) {
                    fileHashes.set(doc.metadata.path, doc.metadata.file_hash);
                }
            });

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const filePath = file.fsPath;
                const currentContent = fs.readFileSync(filePath, 'utf-8');
                const currentHash = crypto.createHash('sha256').update(currentContent).digest('hex');

                if (fileHashes.get(filePath) === currentHash) {
                    continue; // Skip unchanged files (Deterministic + Incremental)
                }

                progress.report({
                    message: `Indexing ${filePath.split('/').pop()} (${i + 1}/${files.length})`,
                    increment: (1 / files.length) * 100
                });
                await this.indexFile(filePath, currentContent, currentHash);
            }
        });
    }

    public async indexFile(filePath: string, content?: string, fileHash?: string): Promise<void> {
        try {
            // Clear existing entries for this file (handles deletions and cleanup)
            await this.bridge.queryMemory('kb_clear_file', { path: filePath });

            if (!content) content = fs.readFileSync(filePath, 'utf-8');
            if (!fileHash) fileHash = crypto.createHash('sha256').update(content).digest('hex');

            const lines = content.split('\n');
            const chunks: string[] = [];
            const metadatas: any[] = [];

            for (let i = 0; i < lines.length; i += 50) {
                const chunk = lines.slice(i, i + 50).join('\n');
                if (chunk.trim()) {
                    const chunkHash = crypto.createHash('sha256').update(chunk).digest('hex');
                    // Stable chunk ID: hash(path + chunk_index + content_hash)
                    const stableId = crypto.createHash('sha256').update(`${filePath}-${i}-${chunkHash}`).digest('hex');

                    chunks.push(chunk);
                    metadatas.push({
                        path: filePath,
                        lineStart: i,
                        file_hash: fileHash,
                        content_hash: chunkHash,
                        stable_id: stableId,
                        timestamp: Date.now()
                    });
                }
            }

            if (chunks.length > 0) {
                const BATCH_SIZE = 20;
                for (let j = 0; j < chunks.length; j += BATCH_SIZE) {
                    await this.bridge.queryMemory('kb_add_batch', {
                        texts: chunks.slice(j, j + BATCH_SIZE),
                        metadatas: metadatas.slice(j, j + BATCH_SIZE)
                    });
                }
            }
        } catch (error) {
            console.error(`Error indexing file ${filePath}:`, error);
        }
    }
}
