import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, StatusBar } from 'react-native';
import { MapViewComponent } from './src/components/MapView';
import { LaneBar } from './src/components/LaneBar';
import { useNavigationStore } from './src/stores/navigationStore';
import { piClient } from './src/services/piClient';
import { locationService } from './src/services/locationService';
import { UserLocation } from './src/services/locationService';

const PI_IP_ADDRESS = '192.168.1.42'; 

const App: React.FC = () => {
  const { 
    setLocation, 
    setPiLaneData, 
    setRecommendedLanes, 
    setPiConnectionStatus,
    currentLocation 
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
  }, []);

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