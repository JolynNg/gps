import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useNavigationStore } from '../stores/navigationStore';

export const LaneBar: React.FC = () => {
  const { piLaneData, recommendedLanes } = useNavigationStore();

  if (!piLaneData) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Waiting for lane data...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Total Lanes: {piLaneData.lane_count}</Text>
      <View style={styles.lanesContainer}>
        {Array.from({ length: piLaneData.lane_count }, (_, i) => (
          <View
            key={i}
            style={[
              styles.lane,
              recommendedLanes.includes(i + 1) && styles.recommendedLane,
            ]}
          >
            <Text style={styles.laneNumber}>{i + 1}</Text>
          </View>
        ))}
      </View>
      <Text style={styles.confidence}>
        Confidence: {(piLaneData.confidence * 100).toFixed(0)}%
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  lanesContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginVertical: 12,
  },
  lane: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#e0e0e0',
    marginHorizontal: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recommendedLane: {
    backgroundColor: '#4CAF50',
  },
  laneNumber: {
    color: '#fff',
    fontWeight: 'bold',
  },
  confidence: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  text: {
    textAlign: 'center',
    color: '#999',
  },
});