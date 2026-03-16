import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { PythonBridge } from "./pythonBridge";

/**
 * MCP Server that exposes Achilles' Knowledge Base as a tool for other AI clients.
 */
export class AchillesMCPServer {
    private server: Server;
    private bridge: PythonBridge;

    constructor(bridge: PythonBridge) {
        this.bridge = bridge;
        this.server = new Server(
            {
                name: "achilles-mcp",
                version: "0.0.1",
            },
            {
                capabilities: {
                    tools: {},
                },
            }
        );

        this.setupHandlers();
    }

    private setupHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
            tools: [
                {
                    name: "query_achilles_memory",
                    description: "Query the Achilles Knowledge Base for project context",
                    inputSchema: {
                        type: "object",
                        properties: {
                            query: { type: "string", description: "The search query" },
                        },
                        required: ["query"],
                    },
                },
            ],
        }));

        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            if (request.params.name === "query_achilles_memory") {
                const query = request.params.arguments?.query as string;
                const results = await this.bridge.queryMemory('kb', { text: query });
                return {
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify(results, null, 2),
                        },
                    ],
                };
            }
            throw new Error("Tool not found");
        });
    }

    public async run() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error("Achilles MCP Server running on stdio");
    }
}
