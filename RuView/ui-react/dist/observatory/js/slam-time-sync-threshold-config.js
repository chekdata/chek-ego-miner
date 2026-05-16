export const SLAM_TIME_SYNC_GATE_V1 = {
  schemaVersion: '1.0.0',
  version: 'slam-time-sync-training-v1',
  status: 'candidate',
  timeSync: {
    minimumFreshOkDevices: 2,
    recommendedFreshOkDevices: 3,
    maxRttMs: 20,
    maxAbsClockOffsetMs: 5,
    okWindowMs: 5000,
  },
  spatial: {
    requireStereoRig: true,
    requirePhonePose: true,
    minHandMatchScore: 0.45,
    maxWristGapM: 0.22,
    maxTargetLatencyMs: 250,
  },
  training: {
    minSuggestedQualityScore: 0.7,
    minRecordingQuality: 0.7,
    maxDomainGapRatio: 1.5,
    minAdaptationSpeedup: 5,
    requireFrozenThresholdsForTrainingReady: true,
  },
};

export function getSlamTimeSyncThresholdPrefix() {
  return SLAM_TIME_SYNC_GATE_V1.status === 'frozen' ? '冻结门槛' : '候选门槛';
}

export function formatObservatoryTimeSyncThresholdDetail() {
  return `${getSlamTimeSyncThresholdPrefix()}：RTT <= ${SLAM_TIME_SYNC_GATE_V1.timeSync.maxRttMs} ms / |offset| <= ${SLAM_TIME_SYNC_GATE_V1.timeSync.maxAbsClockOffsetMs} ms`;
}

export function formatObservatorySpatialThresholdDetail() {
  return `${getSlamTimeSyncThresholdPrefix()}：hand_match >= ${SLAM_TIME_SYNC_GATE_V1.spatial.minHandMatchScore.toFixed(2)} / wrist_gap <= ${(SLAM_TIME_SYNC_GATE_V1.spatial.maxWristGapM * 100).toFixed(1)} cm`;
}
