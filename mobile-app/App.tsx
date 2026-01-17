import React, { useEffect, useState } from 'react';
import { View, StyleSheet, StatusBar } from 'react-native';
import { MapViewComponent } from './src/components/MapView';
import { LaneBar } from './src/components/LaneBar';
import { useNavigationStore } from './src/stores/navigationStore';
import { piClient } from './src/services/piClient';
import { locationService } from './src/services/locationService';

// Update this with your Raspberry Pi's IP address
const PI_IP_ADDRESS = '192.168.1.100'; // Change this!

const App: React.FC = () => {
  const { 
    setLocation, 
    setPiLaneData, 
    setRecommendedLanes, 
    setPiConnectionStatus 
  } = useNavigationStore();
  
  const [hasPermissions, setHasPermissions] = useState(false);

  useEffect(() => {
    // Request location permissions and start tracking
    locationService.requestPermissions().then((granted) => {
      setHasPermissions(granted);
      if (granted) {
        locationService.startTracking((location) => {
          setLocation(location);
        });
      }
    });

    // Connect to Raspberry Pi
    piClient.connect(PI_IP_ADDRESS, (data) => {
      setPiLaneData(data);
      setPiConnectionStatus(true);
      // Use Pi's recommended_lanes directly for now
      // Later: fuse with map/maneuver data
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