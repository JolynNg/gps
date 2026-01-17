//connects to pi to receive lane metadata

export interface LaneMetadata {
  lane_count: number;
  recommended_lanes: number[];
  confidence: number;
  timestamp: number;
  model_version: string;
}

class PiClient {
  private ws: WebSocket | null = null;
  private reconnectInterval: number | null = null;
  private onMessageCallback: ((data: LaneMetadata) => void) | null = null;
  private piAddress: string = '';

  connect(address: string, onMessage: (data: LaneMetadata) => void) {
      this.piAddress = address;
      this.onMessageCallback = onMessage;
      this.attemptConnection();
  }

  private attemptConnection() {
      const wsUrl = `ws://${this.piAddress}:8000/ws/lane-metadata`;
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('Connected to Pi');
        if (this.reconnectInterval) {
          clearInterval(this.reconnectInterval);
          this.reconnectInterval = null;
        }
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data: LaneMetadata = JSON.parse(event.data);
          // Check for stale data (older than 800ms)
          const age = Date.now() - (data.timestamp * 1000);
          if (age < 800 && this.onMessageCallback) {
            this.onMessageCallback(data);
          }
        } catch (e) {
          console.error('Failed to parse Pi message:', e);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('Pi WebSocket error:', error);
      };
      
      this.ws.onclose = () => {
        console.log('Pi connection closed, reconnecting...');
        this.scheduleReconnect();
      };
    }
  
    private scheduleReconnect() {
      if (!this.reconnectInterval) {
        this.reconnectInterval = setInterval(() => {
          this.attemptConnection();
        }, 3000);
      }
    }
  
    disconnect() {
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
      if (this.reconnectInterval) {
        clearInterval(this.reconnectInterval);
        this.reconnectInterval = null;
      }
    }
  }
  
export const piClient = new PiClient();