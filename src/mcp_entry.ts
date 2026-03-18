import { AchillesMCPServer } from "./comms/mcpServer";
import { PythonBridge } from "./comms/pythonBridge";

/**
 * Standalone entry point for the MCP server.
 * This can be run as a separate process to avoid Stdio conflicts with VS Code.
 */
async function main() {
    const bridge = new PythonBridge();
    const mcpServer = new AchillesMCPServer(bridge);
    await mcpServer.run();
}

main().catch(console.error);
