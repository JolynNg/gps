import React, { useEffect, useRef } from 'react';
import { StyleSheet } from 'react-native';
import MapView, { Region } from 'react-native-maps';
import { useNavigationStore } from '../stores/navigationStore';

export const MapViewComponent: React.FC = () => {
  const { currentLocation } = useNavigationStore();
  const mapRef = useRef<MapView>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (currentLocation && mapRef.current) {
      const region: Region = {
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      };

      // Animate to location (only once initially, then let followsUserLocation handle it)
      if (!hasAnimated.current) {
        mapRef.current.animateToRegion(region, 1000);
        hasAnimated.current = true;
      }
    }
  }, [currentLocation]);

  return (
    <MapView
      ref={mapRef}
      style={styles.map}
      initialRegion={{
        latitude: currentLocation?.latitude || 3.1390,  // Use GPS or fallback
        longitude: currentLocation?.longitude || 101.6869,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      }}
      showsUserLocation={true}
      showsMyLocationButton={true}
      followsUserLocation={true}
    />
  );
};

const styles = StyleSheet.create({
  map: {
    ...StyleSheet.absoluteFillObject,
  },
});