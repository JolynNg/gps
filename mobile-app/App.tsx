import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, StatusBar } from 'react-native';
import { MapViewComponent } from './src/components/MapView';
import { LaneBar } from './src/components/LaneBar';
import { useNavigationStore } from './src/stores/navigationStore';
import { piClient } from './src/services/piClient';
import { locationService } from './src/services/locationService';
import { UserLocation } from './src/services/locationService';
import { backendClient } from './src/services/backendClient';

const PI_IP_ADDRESS = '192.168.1.42'; 

// Simple geohash function (basic implementation)
// For production, consider using a library like 'ngeohash'
function generateGeohash(location: UserLocation, precision: number = 8): string {
  const lat = location.latitude;
  const lon = location.longitude;
  
  // Simple geohash encoding (basic implementation)
  // This is a simplified version - consider using a proper library for production
  const chars = '0123456789bcdefghjkmnpqrstuvwxyz';
  let geohash = '';
  let bits = 0;
  let bit = 0;
  let ch = 0;
  let even = true;
  
  let latMin = -90.0, latMax = 90.0;
  let lonMin = -180.0, lonMax = 180.0;
  
  while (geohash.length < precision) {
    if (even) {
      const lonMid = (lonMin + lonMax) / 2;
      if (lon >= lonMid) {
        ch |= (1 << (4 - bit));
        lonMin = lonMid;
      } else {
        lonMax = lonMid;
      }
    } else {
      const latMid = (latMin + latMax) / 2;
      if (lat >= latMid) {
        ch |= (1 << (4 - bit));
        latMin = latMid;
      } else {
        latMax = latMid;
      }
    }
    
    even = !even;
    if (bit < 4) {
      bit++;
    } else {
      geohash += chars[ch];
      bit = 0;
      ch = 0;
    }
  }
  
  return geohash;
}

const App: React.FC = () => {
  const { 
    setLocation, 
    setPiLaneData, 
    setRecommendedLanes, 
    setPiConnectionStatus,
    currentLocation,
    piLaneData  // Added: get piLaneData from store
  } = useNavigationStore();
  
  const [hasPermissions, setHasPermissions] = useState(false);
  const lastLocationRef = useRef<UserLocation | null>(null);

  useEffect(() => {
    // Request location permissions and start tracking
    locationService.requestPermissions().then((granted) => {
      setHasPermissions(granted);
      if (granted) {
        locationService.startTracking((location) => {
          console.log('📍 GPS Location:', location.latitude, location.longitude);
          // Only update if location actually changed
          const lastLoc = lastLocationRef.current;
          if (!lastLoc || 
              Math.abs(lastLoc.latitude - location.latitude) > 0.0001 ||
              Math.abs(lastLoc.longitude - location.longitude) > 0.0001) {
            lastLocationRef.current = location;
            setLocation(location);
          }
        });
      }
    });

    // Connect to Raspberry Pi
    piClient.connect(PI_IP_ADDRESS, (data) => {
      setPiLaneData(data);
      setPiConnectionStatus(true);
      setRecommendedLanes(data.recommended_lanes);
    });

    // Cleanup on unmount
    return () => {
      locationService.stopTracking();
      piClient.disconnect();
    };
  }, []); // Empty deps for initial setup only

  // Separate effect for sending telemetry (depends on location and pi data)
  useEffect(() => {
    if (!currentLocation || !piLaneData) {
      return; // Don't send if we don't have both
    }

    // Send telemetry immediately when we have both location and pi data
    const sendTelemetry = () => {
      try {
        const geohash = generateGeohash(currentLocation);
        const heading = currentLocation.heading || 0;
        const speed = currentLocation.speed || 0;
        
        backendClient.sendTelemetry({
          geohash: geohash,
          heading_bucket: Math.floor(heading / 10) * 10,
          speed_bucket: Math.floor(speed / 10) * 10,
          lane_count_estimate: piLaneData.lane_count,
          recommended_lanes: piLaneData.recommended_lanes,
          confidence: piLaneData.confidence,
          model_version: piLaneData.model_version || "lane_v0.1"
        });
      } catch (error) {
        console.error('Error sending telemetry:', error);
      }
    };

    // Send immediately
    sendTelemetry();

    // Then send periodically (every 5 seconds)
    const telemetryInterval = setInterval(sendTelemetry, 5000);

    return () => {
      clearInterval(telemetryInterval);
    };
  }, [currentLocation, piLaneData]); // Re-run when location or pi data changes

  if (!hasPermissions) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="dark-content" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <MapViewComponent />
      <LaneBar />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default App;