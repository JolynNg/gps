//for sending telemetry and getting lane hints. can be done later if want to test the pi connection first
import axios from 'axios'; //for http requests

const API_BASE_URL = 'http://192.168.1.100:8000'; 

export interface TelemetryData {
  geohash: string;
  heading_bucket: number;
  speed_bucket: number;
  lane_count_estimate: number;
  recommended_lanes: number[];
  confidence: number;
  model_version: string;
}

class BackendClient {
  async sendTelemetry(data: TelemetryData): Promise<void> {
    try {
      await axios.post(`${API_BASE_URL}/api/v1/telemetry`, data); // Fixed: added /api/v1 prefix
      console.log('✅ Telemetry sent:', data.geohash);
    } catch (error) {
      console.error('Failed to send telemetry:', error);
    }
  }

  async getLaneHints(geohash: string, heading?: number) {
    try {
      const params = heading ? { geohash, heading } : { geohash };
      const response = await axios.get(`${API_BASE_URL}/api/v1/lane-hints`, { params }); // Fixed: added /api/v1 prefix
      return response.data;
    } catch (error) {
      console.error('Failed to get lane hints:', error);
      return null;
    }
  }
}

export const backendClient = new BackendClient();