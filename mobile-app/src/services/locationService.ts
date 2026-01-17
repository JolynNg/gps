//provides continuous gps updates for position, heading and speed.
import * as Location from 'expo-location';

export interface UserLocation {
  latitude: number;
  longitude: number;
  heading: number;
  speed: number;
}

class LocationService {
  private watchSubscription: Location.LocationSubscription | null = null;
  private onLocationUpdate: ((location: UserLocation) => void) | null = null;

  async requestPermissions(): Promise<boolean> {
    const { status } = await Location.requestForegroundPermissionsAsync();
    return status === 'granted';
  }

  async startTracking(onUpdate: (location: UserLocation) => void) {
    this.onLocationUpdate = onUpdate;
    
    const hasPermission = await this.requestPermissions();
    if (!hasPermission) {
      console.error('Location permission denied');
      return;
    }

    // Start watching position
    this.watchSubscription = await Location.watchPositionAsync(
      {
        accuracy: Location.Accuracy.High,
        timeInterval: 1000,
        distanceInterval: 10,
      },
      (location) => {
        if (this.onLocationUpdate) {
          this.onLocationUpdate({
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            heading: location.coords.heading || 0,
            speed: location.coords.speed || 0,
          });
        }
      }
    );
  }

  stopTracking() {
    if (this.watchSubscription) {
      this.watchSubscription.remove();
      this.watchSubscription = null;
    }
  }
}

export const locationService = new LocationService();