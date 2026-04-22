/**
 * Claude on Vertex AI Provider Extension for pi
 *
 * Registers Claude models accessed via Google Cloud Vertex AI using
 * Application Default Credentials (ADC) — no API key needed.
 *
 * If you already use Claude Code with Vertex AI (CLAUDE_CODE_USE_VERTEX=1),
 * you're already configured. Just install and go.
 *
 * Setup:
 *   1. cd ~/.pi/agent/extensions/claude-vertex && npm install
 *   2. Ensure these env vars are set (Claude Code sets them automatically):
 *        ANTHROPIC_VERTEX_PROJECT_ID   your GCP project ID
 *        CLOUD_ML_REGION               e.g. us-east5, global
 *   3. Authenticate if needed: gcloud auth application-default login
 *   4. Run pi — select models under the "claude-vertex" provider
 */

import { execSync } from "node:child_process";
import AnthropicVertex from "@anthropic-ai/vertex-sdk";
import type { ContentBlockParam, MessageCreateParamsStreaming } from "@anthropic-ai/sdk/resources/messages.js";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  calculateCost,
  createAssistantMessageEventStream,
  type ImageContent,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type StopReason,
  type TextContent,
  type ThinkingContent,
  type Tool,
  type ToolCall,
  type ToolResultMessage,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// =============================================================================
// Message / Tool Conversion (same as Anthropic direct, no OAuth stealth needed)
// =============================================================================

function sanitizeSurrogates(text: string): string {
  return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

function convertContentBlocks(
  content: (TextContent | ImageContent)[],
): string | Array<{ type: "text"; text: string } | { type: "image"; source: any }> {
  const hasImages = content.some((c) => c.type === "image");
  if (!hasImages) {
    return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join("\n"));
  }

  const blocks = content.map((block) => {
    if (block.type === "text") {
      return { type: "text" as const, text: sanitizeSurrogates(block.text) };
    }
    return {
      type: "image" as const,
      source: {
        type: "base64" as const,
        media_type: block.mimeType,
        data: block.data,
      },
    };
  });

  if (!blocks.some((b) => b.type === "text")) {
    blocks.unshift({ type: "text" as const, text: "(see attached image)" });
  }

  return blocks;
}

function convertMessages(messages: Message[], tools?: Tool[]): any[] {
  const params: any[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    // Skip aborted assistant messages: if this assistant message contains toolCall
    // blocks but the next message(s) are not toolResult, the tool_use blocks have
    // no matching tool_result and Vertex will return a 400. Drop the whole message.
    if (msg.role === "assistant") {
      const hasToolCalls = (msg.content as any[]).some((b: any) => b.type === "toolCall");
      if (hasToolCalls) {
        const next = messages[i + 1];
        if (!next || next.role !== "toolResult") {
          // Orphaned tool_use blocks — skip this message entirely
          continue;
        }
      }
    }

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        if (msg.content.trim()) {
          params.push({ role: "user", content: sanitizeSurrogates(msg.content) });
        }
      } else {
        const blocks: ContentBlockParam[] = (msg.content as any[]).map((item) =>
          item.type === "text"
            ? { type: "text" as const, text: sanitizeSurrogates(item.text) }
            : {
                type: "image" as const,
                source: { type: "base64" as const, media_type: item.mimeType as any, data: item.data },
              },
        );
        if (blocks.length > 0) {
          params.push({ role: "user", content: blocks });
        }
      }
    } else if (msg.role === "assistant") {
      const blocks: ContentBlockParam[] = [];
      for (const block of msg.content) {
        if (block.type === "text" && block.text.trim()) {
          blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
        } else if (block.type === "thinking") {
          // Redacted thinking: pass the opaque payload back as redacted_thinking
          if (block.redacted) {
            blocks.push({
              type: "redacted_thinking" as any,
              data: block.thinkingSignature || "",
            });
            continue;
          }

          if (!block.thinking || block.thinking.trim().length === 0) {
            continue;
          }

          // If thinking signature is missing/empty, convert to plain text block
          if (!block.thinkingSignature || block.thinkingSignature.trim().length === 0) {
            blocks.push({
              type: "text",
              text: sanitizeSurrogates(block.thinking),
            });
          } else {
            blocks.push({
              type: "thinking" as any,
              thinking: sanitizeSurrogates(block.thinking),
              signature: block.thinkingSignature,
            });
          }
        } else if (block.type === "toolCall") {
          blocks.push({
            type: "tool_use",
            id: block.id,
            name: block.name,
            input: block.arguments,
          });
        }
      }
      if (blocks.length > 0) {
        params.push({ role: "assistant", content: blocks });
      }
    } else if (msg.role === "toolResult") {
      const toolResults: any[] = [];
      toolResults.push({
        type: "tool_result",
        tool_use_id: msg.toolCallId,
        content: convertContentBlocks(msg.content),
        is_error: msg.isError,
      });

      let j = i + 1;
      while (j < messages.length && messages[j].role === "toolResult") {
        const nextMsg = messages[j] as ToolResultMessage;
        toolResults.push({
          type: "tool_result",
          tool_use_id: nextMsg.toolCallId,
          content: convertContentBlocks(nextMsg.content),
          is_error: nextMsg.isError,
        });
        j++;
      }
      i = j - 1;
      params.push({ role: "user", content: toolResults });
    }
  }

  // Prompt caching: mark last user message block
  if (params.length > 0) {
    const last = params[params.length - 1];
    if (last.role === "user" && Array.isArray(last.content)) {
      const lastBlock = last.content[last.content.length - 1];
      if (lastBlock) {
        lastBlock.cache_control = { type: "ephemeral" };
      }
    }
  }

  return params;
}

function convertTools(tools: Tool[]): any[] {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: {
      type: "object",
      properties: (tool.parameters as any).properties || {},
      required: (tool.parameters as any).required || [],
    },
  }));
}

function mapStopReason(reason: string): StopReason {
  switch (reason) {
    case "end_turn":
    case "pause_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "toolUse";
    default:
      return "error";
  }
}

// =============================================================================
// ADC Re-authentication Helper
// =============================================================================

function isReauthError(error: any): boolean {
  const msg = error?.message || error?.error_description || JSON.stringify(error);
  return msg.includes("Reauthentication") || msg.includes("invalid_rapt") || msg.includes("invalid_grant");
}

function refreshADC(): void {
  console.log("[claude-vertex] ADC expired — launching gcloud auth application-default login...");
  execSync("gcloud auth application-default login --quiet", {
    encoding: "utf-8",
    timeout: 120_000,
    stdio: "inherit",
  });
}

// =============================================================================
// Streaming Implementation using AnthropicVertex client
// =============================================================================

function streamClaudeVertex(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    let projectId = "unknown";
    let region = "unknown";
    try {
      // Resolve project & region from env vars (required for Vertex)
      projectId = process.env.ANTHROPIC_VERTEX_PROJECT_ID || "unknown";
      region = process.env.CLOUD_ML_REGION || process.env.GOOGLE_CLOUD_LOCATION || "unknown";

      if (!projectId || projectId === "unknown") {
        throw new Error(
          "ANTHROPIC_VERTEX_PROJECT_ID is not set. " +
          "Set it to your GCP project ID, or enable Claude Code's Vertex mode with CLAUDE_CODE_USE_VERTEX=1.",
        );
      }
      if (!region || region === "unknown") {
        throw new Error(
          "CLOUD_ML_REGION is not set. " +
          "Set it to your Vertex AI region, e.g. us-east5 or global.",
        );
      }

      // AnthropicVertex uses ADC automatically - no API key needed
      const client = new AnthropicVertex({
        projectId,
        region,
      });

      const params: MessageCreateParamsStreaming = {
        model: model.id,
        messages: convertMessages(context.messages, context.tools),
        max_tokens: options?.maxTokens || Math.floor(model.maxTokens / 3),
        stream: true,
      };

      // System prompt with prompt caching
      if (context.systemPrompt) {
        (params as any).system = [
          {
            type: "text",
            text: sanitizeSurrogates(context.systemPrompt),
            cache_control: { type: "ephemeral" },
          },
        ];
      }

      if (context.tools && context.tools.length > 0) {
        params.tools = convertTools(context.tools);
      }

      // Extended thinking / reasoning
      if (options?.reasoning && model.reasoning) {
        // claude-opus-4-7+ uses adaptive thinking with output_config.effort
        // older models use enabled thinking with budget_tokens
        // Detect by model ID since custom properties are stripped by pi's model registry
        const usesAdaptiveThinking = /opus-4[-.]?7/i.test(model.id);

        if (usesAdaptiveThinking) {
          const effortMap: Record<string, string> = {
            minimal: "low",
            low: "low",
            medium: "medium",
            high: "high",
            xhigh: "high",
          };
          const effort = effortMap[options.reasoning] ?? "medium";
          (params as any).thinking = { type: "adaptive" };
          (params as any).output_config = { effort };
        } else {
          const defaultBudgets: Record<string, number> = {
            minimal: 1024,
            low: 4096,
            medium: 10240,
            high: 20480,
            xhigh: 32768,
          };
          const customBudget =
            options.thinkingBudgets?.[options.reasoning as keyof typeof options.thinkingBudgets];
          const budget = customBudget ?? defaultBudgets[options.reasoning] ?? 10240;

          (params as any).thinking = {
            type: "enabled",
            budget_tokens: budget,
          };

          // max_tokens MUST be greater than thinking.budget_tokens
          if (!params.max_tokens || params.max_tokens <= budget) {
            params.max_tokens = budget + 4096; // Give 4k tokens buffer for actual text output
          }
        }
      }

      const anthropicStream = await client.messages.create(
        { ...params } as any,
        options?.signal ? { signal: options.signal } : undefined,
      );

      stream.push({ type: "start", partial: output });

      type Block = (ThinkingContent | TextContent | (ToolCall & { partialJson: string })) & {
        index: number;
      };
      const blocks = output.content as Block[];

      for await (const event of anthropicStream) {
        if (event.type === "message_start") {
          output.usage.input = event.message.usage.input_tokens || 0;
          output.usage.output = event.message.usage.output_tokens || 0;
          output.usage.cacheRead = (event.message.usage as any).cache_read_input_tokens || 0;
          output.usage.cacheWrite = (event.message.usage as any).cache_creation_input_tokens || 0;
          output.usage.totalTokens =
            output.usage.input +
            output.usage.output +
            output.usage.cacheRead +
            output.usage.cacheWrite;
          calculateCost(model, output.usage);
        } else if (event.type === "content_block_start") {
          if (event.content_block.type === "text") {
            output.content.push({ type: "text", text: "", index: event.index } as any);
            stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
          } else if (event.content_block.type === "thinking") {
            output.content.push({
              type: "thinking",
              thinking: "",
              thinkingSignature: "",
              index: event.index,
            } as any);
            stream.push({ type: "thinking_start", contentIndex: output.content.length - 1, partial: output });
          } else if ((event.content_block as any).type === "redacted_thinking") {
            output.content.push({
              type: "thinking",
              thinking: "[Reasoning redacted]",
              thinkingSignature: (event.content_block as any).data,
              redacted: true,
              index: event.index,
            } as any);
            stream.push({ type: "thinking_start", contentIndex: output.content.length - 1, partial: output });
          } else if (event.content_block.type === "tool_use") {
            output.content.push({
              type: "toolCall",
              id: event.content_block.id,
              name: event.content_block.name,
              arguments: {},
              partialJson: "",
              index: event.index,
            } as any);
            stream.push({ type: "toolcall_start", contentIndex: output.content.length - 1, partial: output });
          }
        } else if (event.type === "content_block_delta") {
          const index = blocks.findIndex((b) => b.index === event.index);
          const block = blocks[index];
          if (!block) continue;

          if (event.delta.type === "text_delta" && block.type === "text") {
            block.text += event.delta.text;
            stream.push({ type: "text_delta", contentIndex: index, delta: event.delta.text, partial: output });
          } else if (event.delta.type === "thinking_delta" && block.type === "thinking") {
            block.thinking += event.delta.thinking;
            stream.push({
              type: "thinking_delta",
              contentIndex: index,
              delta: event.delta.thinking,
              partial: output,
            });
          } else if (event.delta.type === "input_json_delta" && block.type === "toolCall") {
            (block as any).partialJson += event.delta.partial_json;
            try {
              block.arguments = JSON.parse((block as any).partialJson);
            } catch {}
            stream.push({
              type: "toolcall_delta",
              contentIndex: index,
              delta: event.delta.partial_json,
              partial: output,
            });
          } else if (event.delta.type === "signature_delta" && block.type === "thinking") {
            block.thinkingSignature = (block.thinkingSignature || "") + (event.delta as any).signature;
          }
        } else if (event.type === "content_block_stop") {
          const index = blocks.findIndex((b) => b.index === event.index);
          const block = blocks[index];
          if (!block) continue;

          delete (block as any).index;
          if (block.type === "text") {
            stream.push({ type: "text_end", contentIndex: index, content: block.text, partial: output });
          } else if (block.type === "thinking") {
            stream.push({ type: "thinking_end", contentIndex: index, content: block.thinking, partial: output });
          } else if (block.type === "toolCall") {
            try {
              block.arguments = JSON.parse((block as any).partialJson);
            } catch {}
            delete (block as any).partialJson;
            stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: output });
          }
        } else if (event.type === "message_delta") {
          if ((event.delta as any).stop_reason) {
            output.stopReason = mapStopReason((event.delta as any).stop_reason);
          }
          output.usage.output = (event.usage as any).output_tokens || output.usage.output;
          output.usage.totalTokens =
            output.usage.input +
            output.usage.output +
            output.usage.cacheRead +
            output.usage.cacheWrite;
          calculateCost(model, output.usage);
        }
      }

      if (options?.signal?.aborted) {
        throw new Error("Request was aborted");
      }

      stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
      stream.end();
    } catch (error: any) {
      // If this is a reauth error, try refreshing ADC and ask user to retry
      if (isReauthError(error)) {
        try {
          refreshADC();
          output.stopReason = "error";
          output.errorMessage = `[claude-vertex] Credentials refreshed. Please retry your last message.`;
          stream.push({ type: "error", reason: output.stopReason, error: output });
          stream.end();
          return;
        } catch {
          output.stopReason = "error";
          output.errorMessage = `Automatic re-authentication failed. Please run manually:\n  gcloud auth application-default login`;
          stream.push({ type: "error", reason: output.stopReason, error: output });
          stream.end();
          return;
        }
      }
      for (const block of output.content) delete (block as any).index;
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = `[Project: ${projectId || "unknown"}, Region: ${region || "unknown"}] ${error instanceof Error ? error.message : JSON.stringify(error)}`;
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
  pi.registerProvider("claude-vertex", {
    api: "claude-vertex-api",
    // baseUrl and apiKey are required by pi's provider validation but unused —
    // all API calls go through the Vertex SDK via streamSimple.
    baseUrl: "https://unused.example.com",
    apiKey: "unused",
    models: [
      {
        id: "claude-opus-4-7",
        name: "Claude Opus 4.7 (Vertex)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
        contextWindow: 1000000,
        maxTokens: 64000,
      },
      {
        id: "claude-opus-4-6",
        name: "Claude Opus 4.6 (Vertex)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
        contextWindow: 1000000,
        maxTokens: 64000,
      },
      {
        id: "claude-sonnet-4-6",
        name: "Claude Sonnet 4.6 (Vertex)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
        contextWindow: 1000000,
        maxTokens: 64000,
      },
      {
        id: "claude-haiku-4-5@20251001",
        name: "Claude Haiku 4.5 (Vertex)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1 },
        contextWindow: 1000000,
        maxTokens: 8192,
      },
    ],
    streamSimple: streamClaudeVertex,
  });
}
