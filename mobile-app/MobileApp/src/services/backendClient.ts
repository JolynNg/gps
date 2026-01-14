//for sending telemetry and getting lane hints. can be done later if want to test the pi connection first
import axios from 'axios';

const API_BASE_URL = 'https://your-backend-url.com'; // Update later

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
      await axios.post(`${API_BASE_URL}/telemetry`, data);
    } catch (error) {
      console.error('Failed to send telemetry:', error);
    }
  }

  async getLaneHints(geohash: string, heading?: number) {
    try {
      const params = heading ? { geohash, heading } : { geohash };
      const response = await axios.get(`${API_BASE_URL}/lane-hints`, { params });
      return response.data;
    } catch (error) {
      console.error('Failed to get lane hints:', error);
      return null;
    }
  }
}

export const backendClient = new BackendClient();