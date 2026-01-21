Code Explanation

1. piClient.ts — Mobile app WebSocket client
    Purpose: Connects phone to Pi server and receives real-time lane data.
    How it works:
    Connection management:
    connect(): Establishes WebSocket to Pi
    Auto-reconnect: If connection drops, retries every 3 seconds
    Stale data filter: Ignores messages older than 800ms
    Data flow:
    Receives JSON from Pi WebSocket
    Parses into LaneMetadata interface
    Calls callback function with fresh data
    TypeScript fix: Uses ReturnType<typeof setInterval> for proper typing