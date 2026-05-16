function resolveUrlObject(base, path) {
  const url = new URL(base, window.location.href);
  const basePath = url.pathname.replace(/\/$/, "");
  const nextPath = path.replace(/^\/+/, "");
  url.pathname = `${basePath}/${nextPath}`.replace(/\/{2,}/g, "/");
  url.hash = "";
  return url;
}

function buildHttpUrl(base, path) {
  const url = resolveUrlObject(base, path);
  url.search = "";
  return url.toString();
}

function buildAuthHeaders(token = "") {
  const headers = {};
  if (token.trim()) {
    headers.Authorization = `Bearer ${token.trim()}`;
  }
  return headers;
}

function buildWsUrl(base, path, token = "", params = {}) {
  const url = resolveUrlObject(base, path);
  if (token.trim()) {
    url.searchParams.set("token", token.trim());
  }
  Object.entries(params).forEach(([key, value]) => {
    if (value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

function deriveSensingWsUrl(sensingBase, explicitWsUrl = "") {
  if (explicitWsUrl.trim()) {
    return explicitWsUrl.trim();
  }
  try {
    const base = new URL(sensingBase.trim(), window.location.href);
    if (base.origin === window.location.origin && base.pathname.startsWith("/sensing")) {
      const wsBase = new URL(window.location.href);
      wsBase.protocol = wsBase.protocol === "https:" ? "wss:" : "ws:";
      wsBase.hostname = window.location.hostname;
      wsBase.port = "18080";
      wsBase.pathname = "/ws/sensing";
      wsBase.search = "";
      wsBase.hash = "";
      return wsBase.toString();
    }
    base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
    base.pathname = "/ws/sensing";
    base.search = "";
    base.hash = "";
    return base.toString();
  } catch {
    return "";
  }
}

function numberOr(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function readMotionLevel(raw) {
  return String(raw || "").toLowerCase();
}

function deriveMotionScore(motionLevel, motionBandPower) {
  const level = readMotionLevel(motionLevel);
  if (level === "active" || level === "moving" || level === "walking") {
    return 120;
  }
  if (level === "present_still" || level === "still" || level === "idle") {
    return 12;
  }
  const power = numberOr(motionBandPower, 0);
  return Math.max(8, Math.min(160, power * 14));
}

function extractCenter(person) {
  if (!person || typeof person !== "object") {
    return { x: 320, y: 240, width: 120, height: 220 };
  }
  const bbox = person.bbox;
  if (bbox && Number.isFinite(Number(bbox.x)) && Number.isFinite(Number(bbox.y))) {
    const width = Math.max(48, numberOr(bbox.width, 120));
    const height = Math.max(96, numberOr(bbox.height, 220));
    return {
      x: numberOr(bbox.x, 0) + width * 0.5,
      y: numberOr(bbox.y, 0) + height * 0.5,
      width,
      height
    };
  }

  const points = Array.isArray(person.keypoints) ? person.keypoints : [];
  const valid = points
    .map((point) => ({
      x: numberOr(point?.x, NaN),
      y: numberOr(point?.y, NaN)
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!valid.length) {
    return { x: 320, y: 240, width: 120, height: 220 };
  }

  const xs = valid.map((point) => point.x);
  const ys = valid.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    x: (minX + maxX) * 0.5,
    y: (minY + maxY) * 0.5,
    width: Math.max(48, maxX - minX),
    height: Math.max(96, maxY - minY)
  };
}

function derivePose(person, motionScore, center) {
  if (center.width > center.height * 1.08) {
    return "lying";
  }
  if (motionScore >= 80) {
    return "walking";
  }
  if (center.height < 150 && center.width > center.height * 0.8) {
    return "crouching";
  }
  return "standing";
}

function asVec3(value) {
  if (!Array.isArray(value) || value.length < 3) {
    return null;
  }
  const x = numberOr(value[0], NaN);
  const y = numberOr(value[1], NaN);
  const z = numberOr(value[2], NaN);
  if (![x, y, z].every(Number.isFinite)) {
    return null;
  }
  return [x, y, z];
}

function asVec3Array(values) {
  return Array.isArray(values)
    ? values.map(asVec3).filter((point) => point !== null)
    : [];
}

function computeAveragePoint(points) {
  if (!points.length) {
    return null;
  }
  const sum = points.reduce(
    (acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]],
    [0, 0, 0]
  );
  return [sum[0] / points.length, sum[1] / points.length, sum[2] / points.length];
}

function cloneJson(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return value;
  }
}

function computeRootPoint(points) {
  const leftHip = points[11];
  const rightHip = points[12];
  if (leftHip && rightHip) {
    return [
      (leftHip[0] + rightHip[0]) * 0.5,
      (leftHip[1] + rightHip[1]) * 0.5,
      (leftHip[2] + rightHip[2]) * 0.5
    ];
  }
  return computeAveragePoint(points);
}

function computeFloorY(points) {
  const candidates = [points[15], points[16], points[13], points[14], points[11], points[12]].filter(Boolean);
  const source = candidates.length ? candidates : points;
  return Math.min(...source.map((point) => point[1]));
}

function measureBodyExtents(points) {
  if (!Array.isArray(points) || points.length === 0) {
    return null;
  }
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const zs = points.map((point) => point[2]);
  return {
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
    depth: Math.max(...zs) - Math.min(...zs),
  };
}

function measureShoulderWidth(points) {
  const leftShoulder = points[5];
  const rightShoulder = points[6];
  if (!leftShoulder || !rightShoulder) {
    return null;
  }
  const dx = leftShoulder[0] - rightShoulder[0];
  const dy = leftShoulder[1] - rightShoulder[1];
  const dz = leftShoulder[2] - rightShoulder[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function measureSegmentLength(points, leftIndex, rightIndex) {
  const left = points[leftIndex];
  const right = points[rightIndex];
  if (!left || !right) {
    return null;
  }
  const dx = left[0] - right[0];
  const dy = left[1] - right[1];
  const dz = left[2] - right[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function isWithinRange(value, min, max) {
  return Number.isFinite(value) && value >= min && value <= max;
}

function isBalancedPair(left, right, maxRatio = 2.0) {
  if (!Number.isFinite(left) || !Number.isFinite(right)) {
    return false;
  }
  const smaller = Math.max(Math.min(left, right), 0.05);
  const larger = Math.max(left, right);
  return (larger / smaller) <= maxRatio;
}

function measureHumanSegments(points) {
  return {
    shoulder: measureSegmentLength(points, 5, 6),
    hip: measureSegmentLength(points, 11, 12),
    torsoLeft: measureSegmentLength(points, 5, 11),
    torsoRight: measureSegmentLength(points, 6, 12),
    upperArmLeft: measureSegmentLength(points, 5, 7),
    upperArmRight: measureSegmentLength(points, 6, 8),
    forearmLeft: measureSegmentLength(points, 7, 9),
    forearmRight: measureSegmentLength(points, 8, 10),
    thighLeft: measureSegmentLength(points, 11, 13),
    thighRight: measureSegmentLength(points, 12, 14),
    shinLeft: measureSegmentLength(points, 13, 15),
    shinRight: measureSegmentLength(points, 14, 16),
  };
}

function isPlausibleWorldBody(points) {
  const extents = measureBodyExtents(points);
  if (!extents) {
    return false;
  }
  const shoulderWidth = measureShoulderWidth(points);
  const segments = measureHumanSegments(points);
  if (extents.height < 0.45 || extents.height > 2.6) {
    return false;
  }
  if (extents.width > 2.5 || extents.depth > 2.5) {
    return false;
  }
  if (Number.isFinite(shoulderWidth) && (shoulderWidth < 0.08 || shoulderWidth > 1.1)) {
    return false;
  }
  if (!isWithinRange(segments.hip, 0.08, 0.6)) {
    return false;
  }
  if (!isWithinRange(segments.torsoLeft, 0.18, 0.95) || !isWithinRange(segments.torsoRight, 0.18, 0.95)) {
    return false;
  }
  if (!isWithinRange(segments.upperArmLeft, 0.1, 0.6) || !isWithinRange(segments.upperArmRight, 0.1, 0.6)) {
    return false;
  }
  if (!isWithinRange(segments.forearmLeft, 0.08, 0.6) || !isWithinRange(segments.forearmRight, 0.08, 0.6)) {
    return false;
  }
  if (!isWithinRange(segments.thighLeft, 0.12, 0.8) || !isWithinRange(segments.thighRight, 0.12, 0.8)) {
    return false;
  }
  if (!isWithinRange(segments.shinLeft, 0.12, 0.8) || !isWithinRange(segments.shinRight, 0.12, 0.8)) {
    return false;
  }
  if (!isBalancedPair(segments.torsoLeft, segments.torsoRight, 1.8)) {
    return false;
  }
  if (!isBalancedPair(segments.upperArmLeft, segments.upperArmRight, 2.0)) {
    return false;
  }
  if (!isBalancedPair(segments.forearmLeft, segments.forearmRight, 2.0)) {
    return false;
  }
  if (!isBalancedPair(segments.thighLeft, segments.thighRight, 2.0)) {
    return false;
  }
  if (!isBalancedPair(segments.shinLeft, segments.shinRight, 2.0)) {
    return false;
  }
  return true;
}

function degToRad(value) {
  return numberOr(value, 0) * Math.PI / 180;
}

function rotateVec3XYZ(point, rotationDeg = [0, 0, 0]) {
  const rx = degToRad(rotationDeg[0]);
  const ry = degToRad(rotationDeg[1]);
  const rz = degToRad(rotationDeg[2]);
  let [x, y, z] = point;

  const cosX = Math.cos(rx);
  const sinX = Math.sin(rx);
  let nextY = y * cosX - z * sinX;
  let nextZ = y * sinX + z * cosX;
  y = nextY;
  z = nextZ;

  const cosY = Math.cos(ry);
  const sinY = Math.sin(ry);
  let nextX = x * cosY + z * sinY;
  nextZ = -x * sinY + z * cosY;
  x = nextX;
  z = nextZ;

  const cosZ = Math.cos(rz);
  const sinZ = Math.sin(rz);
  nextX = x * cosZ - y * sinZ;
  nextY = x * sinZ + y * cosZ;
  x = nextX;
  y = nextY;

  return [x, y, z];
}

function buildBasisMatrix(right, up, forward) {
  if (![right, up, forward].every((vector) => Array.isArray(vector) && vector.length >= 3)) {
    return null;
  }
  return [
    [numberOr(right[0], 0), numberOr(up[0], 0), numberOr(forward[0], 0)],
    [numberOr(right[1], 0), numberOr(up[1], 0), numberOr(forward[1], 0)],
    [numberOr(right[2], 0), numberOr(up[2], 0), numberOr(forward[2], 0)],
  ];
}

function transpose3(matrix) {
  return [
    [matrix[0][0], matrix[1][0], matrix[2][0]],
    [matrix[0][1], matrix[1][1], matrix[2][1]],
    [matrix[0][2], matrix[1][2], matrix[2][2]],
  ];
}

function multiplyMat3(a, b) {
  const out = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      out[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
    }
  }
  return out;
}

function multiplyMat3Vec3(matrix, vector) {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  ];
}

function hasSceneRoomTransform(sceneGeometry) {
  const geometry = sceneGeometry?.geometry;
  return Boolean(
    sceneGeometry?.exists
      && geometry?.stereo_rig
      && Array.isArray(geometry.stereo_rig.position_m)
      && geometry.stereo_rig.position_m.length >= 3
      && Array.isArray(geometry.stereo_rig.rotation_deg)
      && geometry.stereo_rig.rotation_deg.length >= 3
  );
}

function transformStereoPointsToRoomWorld(points, sceneGeometry) {
  if (!hasSceneRoomTransform(sceneGeometry)) {
    return null;
  }
  const stereoRig = sceneGeometry.geometry.stereo_rig;
  const translation = stereoRig.position_m.map((value) => numberOr(value, 0));
  const rotationDeg = stereoRig.rotation_deg.map((value) => numberOr(value, 0));
  return points.map((point) => {
    const rotated = rotateVec3XYZ(point, rotationDeg);
    return [
      rotated[0] + translation[0],
      rotated[1] + translation[1],
      rotated[2] + translation[2],
    ];
  });
}

function estimatePoseFromKeypoints(points) {
  if (!points.length) {
    return "standing";
  }
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const zs = points.map((point) => point[2]);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const depth = Math.max(...zs) - Math.min(...zs);

  if (height < 0.95 && Math.max(width, depth) > 0.7) {
    return "lying";
  }
  if (height < 1.25) {
    return "crouching";
  }
  return "standing";
}

function squaredDistanceXZ(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length < 3 || right.length < 3) {
    return Number.POSITIVE_INFINITY;
  }
  const dx = numberOr(left[0], NaN) - numberOr(right[0], NaN);
  const dz = numberOr(left[2], NaN) - numberOr(right[2], NaN);
  if (![dx, dz].every(Number.isFinite)) {
    return Number.POSITIVE_INFINITY;
  }
  return dx * dx + dz * dz;
}

function midpoint3(left, right) {
  if (!left || !right) {
    return null;
  }
  return [
    (left[0] + right[0]) * 0.5,
    (left[1] + right[1]) * 0.5,
    (left[2] + right[2]) * 0.5,
  ];
}

function subtract3(left, right) {
  if (!left || !right) {
    return null;
  }
  return [
    left[0] - right[0],
    left[1] - right[1],
    left[2] - right[2],
  ];
}

function scale3(vector, scalar) {
  if (!vector) {
    return null;
  }
  return [
    vector[0] * scalar,
    vector[1] * scalar,
    vector[2] * scalar,
  ];
}

function dot3(left, right) {
  if (!left || !right) {
    return 0;
  }
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function normalize3(vector) {
  if (!Array.isArray(vector) || vector.length < 3) {
    return null;
  }
  const length = Math.sqrt(
    vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
  );
  if (!Number.isFinite(length) || length < 1e-6) {
    return null;
  }
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function cross3(left, right) {
  if (!left || !right) {
    return null;
  }
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function projectOntoPlane(vector, normal) {
  if (!vector || !normal) {
    return null;
  }
  const factor = dot3(vector, normal);
  return subtract3(vector, scale3(normal, factor));
}

function blendDirection(previous, next, alpha = 0.35) {
  if (!next) {
    return previous || null;
  }
  if (!previous) {
    return next;
  }
  let alignedNext = next;
  if (dot3(previous, next) < 0) {
    alignedNext = scale3(next, -1);
  }
  return normalize3([
    previous[0] * (1 - alpha) + alignedNext[0] * alpha,
    previous[1] * (1 - alpha) + alignedNext[1] * alpha,
    previous[2] * (1 - alpha) + alignedNext[2] * alpha,
  ]);
}

function blendPoint3(previous, next, alpha = 0.35) {
  if (!next) {
    return previous || null;
  }
  if (!previous) {
    return next;
  }
  return [
    previous[0] * (1 - alpha) + next[0] * alpha,
    previous[1] * (1 - alpha) + next[1] * alpha,
    previous[2] * (1 - alpha) + next[2] * alpha,
  ];
}

function addScaled3(origin, vector, scale) {
  if (!origin || !vector) {
    return null;
  }
  return [
    origin[0] + vector[0] * scale,
    origin[1] + vector[1] * scale,
    origin[2] + vector[2] * scale,
  ];
}

function estimateWearerPhonePose(person, previousPose = null) {
  const points = asVec3Array(person?.raw_keypoints);
  if (points.length < 17) {
    return null;
  }
  const leftShoulder = points[5];
  const rightShoulder = points[6];
  const leftHip = points[11];
  const rightHip = points[12];
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
    return null;
  }

  const shoulderMid = midpoint3(leftShoulder, rightShoulder);
  const hipMid = midpoint3(leftHip, rightHip);
  const up = normalize3(subtract3(shoulderMid, hipMid));
  const shoulderRight = normalize3(subtract3(rightShoulder, leftShoulder));
  let forward = normalize3(cross3(shoulderRight, up));
  if (!shoulderMid || !up || !shoulderRight || !forward) {
    return null;
  }

  const nose = points[0] || null;
  const faceVector = nose ? subtract3(nose, shoulderMid) : null;
  const faceForward = normalize3(projectOntoPlane(faceVector, up));
  if (faceForward && dot3(faceForward, forward) < 0) {
    forward = scale3(forward, -1);
  }

  let blendedForward = normalize3([
    forward[0] * 0.4 + (faceForward?.[0] || forward[0]) * 0.6,
    forward[1] * 0.4 + (faceForward?.[1] || forward[1]) * 0.6,
    forward[2] * 0.4 + (faceForward?.[2] || forward[2]) * 0.6,
  ]) || forward;
  if (previousPose?.forward_vector) {
    blendedForward = blendDirection(previousPose.forward_vector, blendedForward, 0.35) || blendedForward;
  }

  let right = normalize3(cross3(up, blendedForward)) || shoulderRight;
  if (dot3(right, shoulderRight) < 0) {
    blendedForward = scale3(blendedForward, -1);
    right = scale3(right, -1);
  }
  right = blendDirection(previousPose?.right_vector || null, right, 0.35) || right;
  const stabilizedUp = blendDirection(previousPose?.up_vector || null, up, 0.25) || up;

  const torsoLength = Math.max(
    Math.sqrt(dot3(subtract3(shoulderMid, hipMid), subtract3(shoulderMid, hipMid))),
    0.2
  );
  const chestDown = Math.min(Math.max(torsoLength * 0.22, 0.08), 0.18);
  const chestForward = Math.min(Math.max(torsoLength * 0.14, 0.05), 0.12);

  const chestCenter = addScaled3(
    addScaled3(shoulderMid, stabilizedUp, -chestDown),
    blendedForward,
    chestForward
  );
  if (!chestCenter) {
    return null;
  }

  const position = blendPoint3(previousPose?.position_m || null, chestCenter, 0.35) || chestCenter;

  return {
    id: `wearer-phone-${person?.id || "unknown"}`,
    label: "手机",
    position_m: position,
    target_space: "room_world_frame",
    right_vector: right,
    up_vector: stabilizedUp,
    forward_vector: blendedForward,
    source: faceForward ? "wearer_body_face_estimate" : "wearer_body_estimate",
    confidence: numberOr(person?.confidence, 0),
  };
}

function normalizeEdgePhonePose(phoneDebug, sceneGeometry) {
  if (!phoneDebug?.fresh || !phoneDebug?.device_pose) {
    return null;
  }
  const pose = cloneJson(phoneDebug.device_pose);
  if (!Array.isArray(pose?.position_m) || pose.position_m.length < 3) {
    return null;
  }
  pose.position_m = pose.position_m.map((value) => numberOr(value, 0));
  const targetSpace = String(pose.target_space || "").trim();
  if (targetSpace === "stereo_pair_frame" && hasSceneRoomTransform(sceneGeometry)) {
    const transformed = transformStereoPointsToRoomWorld([pose.position_m], sceneGeometry);
    if (Array.isArray(transformed) && transformed[0]) {
      pose.position_m = transformed[0];
    }
    const stereoRotationDeg = sceneGeometry?.geometry?.stereo_rig?.rotation_deg || [0, 0, 0];
    ["right_vector", "up_vector", "forward_vector"].forEach((key) => {
      if (Array.isArray(pose[key]) && pose[key].length >= 3) {
        pose[key] = rotateVec3XYZ(pose[key].map((value) => numberOr(value, 0)), stereoRotationDeg);
      }
    });
    pose.target_space = "room_world_frame";
  }
  if (Array.isArray(pose.rotation_deg)) {
    pose.rotation_deg = pose.rotation_deg.map((value) => numberOr(value, 0));
  }
  if (typeof pose.source !== "string" || !pose.source.trim()) {
    pose.source = "edge_device_pose";
  }
  return pose;
}

function isOrientationOnlyPhonePose(pose) {
  if (!pose) {
    return false;
  }
  const source = String(pose.source || "").trim();
  const targetSpace = String(pose.target_space || "").trim();
  const position = Array.isArray(pose.position_m) ? pose.position_m : null;
  const nearOrigin = Boolean(position)
    && position.length >= 3
    && Math.abs(numberOr(position[0], 0)) < 0.001
    && Math.abs(numberOr(position[1], 0)) < 0.001
    && Math.abs(numberOr(position[2], 0)) < 0.001;
  return targetSpace === "device_motion_reference_frame"
    || source.includes("attitude")
    || nearOrigin;
}

function mergeEdgePhonePoseWithEstimate(edgePose, estimatedPose) {
  if (!edgePose) {
    return estimatedPose || null;
  }
  if (!estimatedPose) {
    return edgePose;
  }
  if (!isOrientationOnlyPhonePose(edgePose)) {
    return edgePose;
  }
  return {
    ...cloneJson(estimatedPose),
    rotation_deg: Array.isArray(edgePose.rotation_deg) ? cloneJson(edgePose.rotation_deg) : estimatedPose.rotation_deg,
    right_vector: Array.isArray(edgePose.right_vector) ? cloneJson(edgePose.right_vector) : estimatedPose.right_vector,
    up_vector: Array.isArray(edgePose.up_vector) ? cloneJson(edgePose.up_vector) : estimatedPose.up_vector,
    forward_vector: Array.isArray(edgePose.forward_vector) ? cloneJson(edgePose.forward_vector) : estimatedPose.forward_vector,
    source: `${edgePose.source || "edge_device_pose"}+${estimatedPose.source || "wearer_body_estimate"}`,
    confidence: Math.max(numberOr(edgePose.confidence, 0), numberOr(estimatedPose.confidence, 0)),
  };
}

function chooseReplacementPersonIndex(persons, wearerPerson) {
  if (!Array.isArray(persons) || persons.length === 0 || !wearerPerson?.position) {
    return -1;
  }
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  persons.forEach((person, index) => {
    const distance = squaredDistanceXZ(person?.position, wearerPerson.position);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function isStereoPersonTrackId(trackId) {
  return /^stereo-person-\d+$/i.test(String(trackId || "").trim());
}

function resolveWearerTrackId(edgeOverlay, stereoTrackedPersons = [], rememberedWearerTrackId = "", livePreview = null) {
  const stereoIds = new Set(
    (Array.isArray(stereoTrackedPersons) ? stereoTrackedPersons : [])
      .map((person) => String(person?.id || "").trim())
      .filter(Boolean)
  );
  const liveTargetStereoTrackId = String(
    livePreview?.target_human_state?.target_track_ids?.stereo_track_id || ""
  ).trim();
  const candidates = [
    liveTargetStereoTrackId,
    livePreview?.scene_state?.highlight_target_person_id,
    edgeOverlay?.association?.iphone_operator_track_id,
    edgeOverlay?.association?.selected_operator_track_id,
    edgeOverlay?.association?.stereo_operator_track_id,
  ];
  for (const candidate of candidates) {
    const normalized = String(candidate || "").trim();
    if (!normalized) {
      continue;
    }
    if (stereoIds.size > 0) {
      if (stereoIds.has(normalized)) {
        return normalized;
      }
      continue;
    }
    if (isStereoPersonTrackId(normalized)) {
      return normalized;
    }
  }
  const remembered = String(rememberedWearerTrackId || "").trim();
  if (remembered && stereoIds.has(remembered)) {
    return remembered;
  }
  return "";
}

function buildTrackedPersonFromPoints(bodyPoints, fallbackPerson, motionScore, sourceLabel, isWearer = false) {
  const points = asVec3Array(bodyPoints);
  if (points.length < 17) {
    return null;
  }

  const root = computeRootPoint(points);
  if (!root) {
    return null;
  }

  const floorY = computeFloorY(points);
  const rawKeypoints = points.map((point) => ([
    point[0],
    Math.max(0, point[1] - floorY),
    point[2]
  ]));
  const rootY = Math.max(0, root[1] - floorY);
  const pose = estimatePoseFromKeypoints(rawKeypoints);
  const plausibleRawKeypoints = isPlausibleWorldBody(rawKeypoints);

  if (!plausibleRawKeypoints) {
    return {
      id: fallbackPerson?.id || "operator-fallback",
      position: [root[0], rootY, root[2]],
      motion_score: motionScore,
      pose: fallbackPerson?.pose || pose,
      facing: 0,
      confidence: numberOr(fallbackPerson?.confidence, 0),
      isWearer: Boolean(isWearer),
      source: `${sourceLabel}:implausible-raw`,
    };
  }

  return {
    id: fallbackPerson?.id || "operator-fallback",
    position: [root[0], rootY, root[2]],
    motion_score: motionScore,
    pose,
    facing: 0,
    confidence: numberOr(fallbackPerson?.confidence, 0),
    raw_keypoints: rawKeypoints,
    isWearer: Boolean(isWearer),
    source: sourceLabel,
  };
}

function buildKeypointPerson(bodyPoints, fallbackPerson, motionScore, edgeOverlay, wearerTrackId = "") {
  const points = asVec3Array(bodyPoints);
  if (points.length < 17) {
    return null;
  }

  const root = computeRootPoint(points);
  if (!root) {
    return null;
  }

  const floorY = computeFloorY(points);
  const rawKeypoints = points.map((point) => ([
    point[0],
    Math.max(0, point[1] - floorY),
    point[2]
  ]));
  const rootY = Math.max(0, root[1] - floorY);
  const pose = estimatePoseFromKeypoints(rawKeypoints);
  const plausibleRawKeypoints = isPlausibleWorldBody(rawKeypoints);

  const selectedTrackId = String(edgeOverlay?.association?.selected_operator_track_id || "").trim();
  const resolvedTrackId = wearerTrackId
    || (isStereoPersonTrackId(selectedTrackId) ? selectedTrackId : "")
    || String(fallbackPerson?.id || "").trim()
    || "operator-main";
  const confirmedWearer = wearerTrackId.length > 0 && resolvedTrackId === wearerTrackId;

  if (!plausibleRawKeypoints) {
    return {
      id: resolvedTrackId,
      position: [root[0], rootY, root[2]],
      motion_score: motionScore,
      pose: fallbackPerson?.pose || pose,
      facing: 0,
      confidence: numberOr(
        edgeOverlay?.fusion?.confidence ?? edgeOverlay?.fusion?.body_conf ?? fallbackPerson?.confidence,
        0
      ),
      isWearer: confirmedWearer,
      source: `wearer-fallback:implausible-raw:${edgeOverlay?.fusion?.body_source || edgeOverlay?.fusion?.selected_source || "edge-overlay"}`,
    };
  }

  return {
    id: resolvedTrackId,
    position: [root[0], rootY, root[2]],
    motion_score: motionScore,
    pose,
    facing: 0,
    confidence: numberOr(
      edgeOverlay?.fusion?.confidence ?? edgeOverlay?.fusion?.body_conf ?? fallbackPerson?.confidence,
      0
    ),
    raw_keypoints: rawKeypoints,
    isWearer: confirmedWearer,
    source: edgeOverlay?.fusion?.body_source || edgeOverlay?.fusion?.selected_source || "edge-overlay"
  };
}

function deriveEdgeStereoPersons(
  edgeOverlay,
  motionScore,
  sceneGeometry = null,
  wearerTrackId = "",
  livePreview = null
) {
  const liveStereoPersons = Array.isArray(livePreview?.scene_state?.tracked_people?.stereo_people)
    ? livePreview.scene_state.tracked_people.stereo_people
    : [];
  const stereoPersons = liveStereoPersons.length > 0
    ? liveStereoPersons
    : (Array.isArray(edgeOverlay?.stereo?.persons) ? edgeOverlay.stereo.persons : []);
  return stereoPersons
    .map((person, index) => {
      let bodyPoints = asVec3Array(person?.body_kpts_3d);
      let bodySpace = String(
        person?.body_space
        || edgeOverlay?.stereo?.body_space
        || ""
      ).trim();
      if (bodySpace === "stereo_pair_frame") {
        const transformed = transformStereoPointsToRoomWorld(bodyPoints, sceneGeometry);
        if (Array.isArray(transformed) && transformed.length >= 17) {
          bodyPoints = transformed;
          bodySpace = "room_world_frame";
        }
      }

      const trackId = String(person?.operator_track_id || `stereo-person-${index}`);
      return buildTrackedPersonFromPoints(
        bodyPoints,
        {
          id: trackId,
          confidence: numberOr(person?.confidence, 0),
        },
        motionScore,
        `stereo:${bodySpace || "unknown"}`,
        wearerTrackId.length > 0 && trackId === wearerTrackId
      );
    })
    .filter(Boolean);
}

function isWorldComparableBodySpace(space, sceneGeometry = null) {
  const normalized = String(space || "").trim();
  if (normalized === "operator_frame" || normalized === "room_world_frame") {
    return true;
  }
  if (normalized === "stereo_pair_frame") {
    return hasSceneRoomTransform(sceneGeometry);
  }
  return false;
}

function resolvePreferredBodyCandidate(edgeOverlay, sceneGeometry = null) {
  const sources = [
    {
      points: edgeOverlay?.fusion?.body_kpts_3d,
      bodySpace: edgeOverlay?.fusion?.body_space,
      source: edgeOverlay?.fusion?.body_source || edgeOverlay?.fusion?.selected_source || "fusion",
    },
    {
      points: edgeOverlay?.stereo?.body_kpts_3d,
      bodySpace: edgeOverlay?.stereo?.body_space,
      source: "stereo",
    },
    {
      points: edgeOverlay?.wifi?.body_kpts_3d,
      bodySpace: edgeOverlay?.wifi?.body_space,
      source: "wifi",
    },
    {
      points: edgeOverlay?.iphone?.body_kpts_3d,
      bodySpace: edgeOverlay?.iphone?.body_space,
      source: "iphone",
    }
  ];
  for (const candidate of sources) {
    const points = asVec3Array(candidate.points);
    let resolvedPoints = points;
    let resolvedSpace = String(candidate.bodySpace || "").trim();
    if (resolvedSpace === "stereo_pair_frame") {
      const transformed = transformStereoPointsToRoomWorld(points, sceneGeometry);
      if (Array.isArray(transformed) && transformed.length >= 17) {
        resolvedPoints = transformed;
        resolvedSpace = "room_world_frame";
      }
    }
    if (resolvedPoints.length >= 17 && isPlausibleWorldBody(resolvedPoints)) {
      return {
        points: resolvedPoints,
        bodySpace: resolvedSpace,
        source: candidate.source,
      };
    }
  }
  return null;
}

function deriveObservatoryPersons(posePayload, sensingPayload) {
  const rawPersons = Array.isArray(posePayload?.persons) ? posePayload.persons : [];
  const features = sensingPayload?.features || {};
  const classification = sensingPayload?.classification || {};
  const motionScore = deriveMotionScore(classification.motion_level, features.motion_band_power);
  const modelLoaded = Boolean(posePayload?.model_status?.loaded || sensingPayload?.model_status?.loaded);

  if (!rawPersons.length) {
    if (modelLoaded) {
      return [];
    }
    if (!classification.presence) {
      return [];
    }
    return [{
      id: "p0",
      position: [0, 0, 0],
      motion_score: motionScore,
      pose: motionScore >= 80 ? "walking" : "standing",
      facing: 0
    }];
  }

  const centers = rawPersons.map(extractCenter);
  const xs = centers.map((center) => center.x);
  const ys = centers.map((center) => center.y);
  const minX = rawPersons.length > 1 ? Math.min(...xs) : xs[0] - 160;
  const maxX = rawPersons.length > 1 ? Math.max(...xs) : xs[0] + 160;
  const minY = rawPersons.length > 1 ? Math.min(...ys) : ys[0] - 140;
  const maxY = rawPersons.length > 1 ? Math.max(...ys) : ys[0] + 140;
  const spanX = Math.max(160, maxX - minX);
  const spanY = Math.max(140, maxY - minY);

  return rawPersons.map((person, index) => {
    const center = centers[index];
    const x = ((center.x - minX) / spanX - 0.5) * 8.4;
    const z = ((center.y - minY) / spanY - 0.5) * 6.2;
    const pose = derivePose(person, motionScore, center);
    return {
      id: person.id || person.person_id || `p${index}`,
      position: [x, pose === "lying" ? 0.45 : 0, z],
      motion_score: motionScore,
      pose,
      facing: 0,
      confidence: numberOr(person.confidence, classification.confidence || 0),
      isWearer: false,
      source: "observatory-fallback"
    };
  });
}

function deriveChekPersons(
  posePayload,
  sensingPayload,
  fusion,
  teleop,
  sceneGeometry = null,
  rememberedWearerTrackId = "",
  livePreview = null
) {
  const observatoryFallbackPersons = deriveObservatoryPersons(posePayload, sensingPayload);
  const features = sensingPayload?.features || {};
  const classification = sensingPayload?.classification || {};
  const motionScore = deriveMotionScore(classification.motion_level, features.motion_band_power);
  const edgeOverlay = buildEdgeOverlay(fusion, teleop);
  const stereoTrackedPersons = deriveEdgeStereoPersons(
    edgeOverlay,
    motionScore,
    sceneGeometry,
    "",
    livePreview
  );
  const wearerTrackId = resolveWearerTrackId(
    edgeOverlay,
    stereoTrackedPersons,
    rememberedWearerTrackId,
    livePreview
  );
  const fallbackPersons = stereoTrackedPersons.length > 0
    ? stereoTrackedPersons.map((person) => ({
        ...person,
        isWearer: wearerTrackId.length > 0 && person.id === wearerTrackId,
      }))
    : observatoryFallbackPersons;
  const preferredBody = resolvePreferredBodyCandidate(edgeOverlay, sceneGeometry);
  const fallbackWithWearer = fallbackPersons
      .map((person) => ({
        ...person,
        isWearer: wearerTrackId.length > 0 && person.id === wearerTrackId
      }))
      .sort((left, right) => Number(Boolean(right.isWearer)) - Number(Boolean(left.isWearer)));

  // When multiple stereo people are visible but phone->stereo matching is not confirmed,
  // keep the main view on stable stereo tracks instead of injecting a fused/wifi pseudo-wearer.
  if (stereoTrackedPersons.length > 1 && wearerTrackId.length === 0) {
    return fallbackWithWearer.slice(0, 4);
  }

  if (!preferredBody?.points?.length) {
    return fallbackWithWearer.slice(0, 4);
  }

  if (!isWorldComparableBodySpace(preferredBody.bodySpace, sceneGeometry)) {
    return fallbackWithWearer
      .map((person) => ({
        ...person,
        source: person.isWearer ? `wearer-fallback:${preferredBody.source}:${preferredBody.bodySpace || "unknown"}` : person.source
      }))
      .slice(0, 4);
  }

  const wearerPerson = buildKeypointPerson(preferredBody.points, fallbackPersons[0], motionScore, edgeOverlay, wearerTrackId);
  if (!wearerPerson) {
    return fallbackWithWearer.slice(0, 4);
  }
  wearerPerson.source = preferredBody.source;

  const mergedPersons = [...fallbackWithWearer];
  const replaceIndex = chooseReplacementPersonIndex(mergedPersons, wearerPerson);
  if (replaceIndex >= 0) {
    mergedPersons[replaceIndex] = wearerPerson;
  } else {
    mergedPersons.unshift(wearerPerson);
  }

  const deduped = [];
  const seenIds = new Set();
  for (const person of mergedPersons) {
    const personId = String(person?.id || "");
    if (personId && seenIds.has(personId)) {
      continue;
    }
    if (personId) {
      seenIds.add(personId);
    }
    deduped.push({
      ...person,
      isWearer: Boolean(person?.isWearer)
    });
  }

  return deduped
    .sort((left, right) => {
      const wearerDelta = Number(Boolean(right?.isWearer)) - Number(Boolean(left?.isWearer));
      if (wearerDelta !== 0) {
        return wearerDelta;
      }
      return numberOr(right?.confidence, 0) - numberOr(left?.confidence, 0);
    })
    .slice(0, 4);
}

function normalizeVitals(vitalSigns) {
  const payload = vitalSigns && typeof vitalSigns === "object" ? { ...vitalSigns } : {};
  payload.breathing_rate_bpm = numberOr(
    payload.breathing_rate_bpm ?? payload.breathing_bpm ?? payload.respiration_bpm,
    0
  );
  payload.heart_rate_bpm = numberOr(payload.heart_rate_bpm ?? payload.heart_bpm, 0);
  return payload;
}

function buildEdgeOverlay(fusion, teleop) {
  const debug = fusion?.operator_debug || {};
  return {
    stereo: debug.stereo_pair || null,
    iphone: debug.iphone_capture || null,
    wifi: debug.wifi_pose || null,
    fusion: debug.fused_pose || null,
    association: debug.association || null,
    motion: debug.motion_state || null,
    teleop: teleop || null
  };
}

async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function summarizeSceneGeometry(geometry) {
  return {
    coordinate_frame_version: geometry?.coordinate_frame_version || null,
    ap_count: Array.isArray(geometry?.ap_nodes) ? geometry.ap_nodes.length : 0,
    stereo_defined: Boolean(geometry?.stereo_rig?.id),
    phone_defined: Boolean(geometry?.phone_pose),
    updated_at_ms: Number.isFinite(Number(geometry?.updated_at_ms))
      ? Number(geometry.updated_at_ms)
      : null,
  };
}

function normalizeSceneGeometryPayload(payload, options = {}) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  if (payload.exists && payload.geometry) {
    return payload;
  }
  if (!payload.geometry) {
    return null;
  }
  return {
    scene_id: payload.scene_id || payload.geometry.scene_id || "",
    scene_name: payload.scene_name || payload.geometry.scene_name || "",
    exists: true,
    auto_draft: Boolean(options.autoDraft),
    ready_to_save: Boolean(payload.ready_to_save),
    confidence_score: numberOr(payload.confidence_score, 0),
    source_breakdown: payload.source_breakdown || null,
    confidence_breakdown: payload.confidence_breakdown || null,
    warnings: Array.isArray(payload.warnings) ? payload.warnings : [],
    summary: summarizeSceneGeometry(payload.geometry),
    geometry: payload.geometry,
  };
}

async function fetchSceneGeometry(edgeBase, authHeaders) {
  const saved = await fetchJson(buildHttpUrl(edgeBase, "/evolution/scene/geometry"), {
    headers: authHeaders,
  }).catch(() => null);
  const normalizedSaved = normalizeSceneGeometryPayload(saved);
  if (normalizedSaved?.exists) {
    return normalizedSaved;
  }

  const draftHeaders = {
    ...authHeaders,
    "Content-Type": "application/json",
  };
  const autoDraft = await fetchJson(
    buildHttpUrl(edgeBase, "/evolution/scene/geometry/auto-draft"),
    {
      method: "POST",
      headers: draftHeaders,
      body: "{}",
    }
  ).catch(() => null);
  return normalizeSceneGeometryPayload(autoDraft, { autoDraft: true }) || normalizedSaved;
}

export function readChekConfigFromLocation() {
  const params = new URLSearchParams(window.location.search);
  const origin = window.location.origin;
  return {
    enabled: params.get("mode") === "chek" || params.has("sensing") || params.has("edge"),
    sensingBase: params.get("sensing") || `${origin}/sensing`,
    sensingWsUrl: params.get("sensingWs") || "",
    edgeBase: params.get("edge") || `${origin}/edge`,
    edgeWsBase: params.get("edgeWs") || "",
    edgeToken: params.get("token") || "",
    stereoPreviewUrl: params.get("stereoPreview") || "/stereo-preview.jpg"
  };
}

export class ChekLiveAdapter {
  constructor(config) {
    this._config = config;
    this._lastMessageAt = 0;
    this._pollTimer = null;
    this._sensingSocket = null;
    this._fusionSocket = null;
    this._teleopSocket = null;
    this._sensing = null;
    this._pose = null;
    this._fusion = null;
    this._teleop = null;
    this._sceneGeometry = null;
    this._livePreview = null;
    this._timeSync = null;
    this._lastWearerPerson = null;
    this._lastWearerPersonAt = 0;
    this._lastConfirmedWearerTrackId = "";
    this._lastConfirmedWearerTrackIdAt = 0;
    this._lastLivePhonePose = null;
    this._lastLivePhonePoseAt = 0;
    this._arkitPhoneRoomAnchor = null;
  }

  get connected() {
    return Date.now() - this._lastMessageAt < 4_000;
  }

  _markActive() {
    this._lastMessageAt = Date.now();
  }

  start() {
    this._connectSensingSocket();
    this._connectEdgeSockets();
    this._pollAll();
    this._pollTimer = window.setInterval(() => this._pollAll(), 1_500);
  }

  stop() {
    if (this._pollTimer !== null) {
      window.clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
    [this._sensingSocket, this._fusionSocket, this._teleopSocket].forEach((socket) => {
      if (socket) {
        socket.close();
      }
    });
  }

  applyExternalSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== "object") {
      return;
    }
    const nextSceneGeometry = snapshot.scene_geometry;
    const nextLivePreview = snapshot.live_preview;
    const nextTimeSync = snapshot.time_sync;
    if (nextSceneGeometry && typeof nextSceneGeometry === "object") {
      this._sceneGeometry = nextSceneGeometry;
    }
    if (nextLivePreview && typeof nextLivePreview === "object") {
      this._livePreview = nextLivePreview;
    }
    if (nextTimeSync && typeof nextTimeSync === "object") {
      this._timeSync = nextTimeSync;
    }
    if (this._sceneGeometry || this._livePreview || this._timeSync) {
      this._markActive();
    }
  }

  _connectSensingSocket() {
    const url = deriveSensingWsUrl(this._config.sensingBase, this._config.sensingWsUrl);
    if (!url) {
      return;
    }

    try {
      this._sensingSocket = new WebSocket(url);
      this._sensingSocket.onmessage = (event) => {
        try {
          const payload = JSON.parse(String(event.data));
          if (payload && payload.type === "pose_data" && payload.payload?.pose) {
            this._pose = payload.payload.pose;
          } else {
            this._sensing = payload;
          }
          this._markActive();
        } catch {}
      };
    } catch {}
  }

  _connectEdgeSockets() {
    if (!this._config.edgeWsBase) {
      return;
    }

    const fusionUrl = buildWsUrl(
      this._config.edgeWsBase,
      "/stream/fusion",
      this._config.edgeToken,
      { debug_views: "operator" }
    );
    const teleopUrl = buildWsUrl(
      this._config.edgeWsBase,
      "/stream/teleop",
      this._config.edgeToken
    );

    try {
      this._fusionSocket = new WebSocket(fusionUrl);
      this._fusionSocket.onmessage = (event) => {
        try {
          this._fusion = JSON.parse(String(event.data));
          this._markActive();
        } catch {}
      };
    } catch {}

    try {
      this._teleopSocket = new WebSocket(teleopUrl);
      this._teleopSocket.onmessage = (event) => {
        try {
          this._teleop = JSON.parse(String(event.data));
          this._markActive();
        } catch {}
      };
    } catch {}
  }

  async _pollAll() {
    const sensingBase = this._config.sensingBase;
    const edgeBase = this._config.edgeBase;
    const authHeaders = buildAuthHeaders(this._config.edgeToken);
    try {
      const sensingLatestPromise = sensingBase
        ? fetchJson(buildHttpUrl(sensingBase, "/api/v1/sensing/latest")).catch(() => null)
        : Promise.resolve(null);
      const poseCurrentPromise = sensingBase
        ? fetchJson(buildHttpUrl(sensingBase, "/api/v1/pose/current")).catch(() => null)
        : Promise.resolve(null);
      const sceneGeometryPromise = edgeBase
        ? fetchSceneGeometry(edgeBase, authHeaders)
        : Promise.resolve(null);
      const livePreviewPromise = edgeBase
        ? fetchJson(buildHttpUrl(edgeBase, "/live-preview.json"), {
          headers: authHeaders,
        }).catch(() => null)
        : Promise.resolve(null);
      const timeSyncPromise = edgeBase
        ? fetchJson(buildHttpUrl(edgeBase, "/time/sync/current"), {
          headers: authHeaders,
        }).catch(() => null)
        : Promise.resolve(null);
      const [latest, poseCurrent, sceneGeometry, livePreview, timeSync] = await Promise.all([
        sensingLatestPromise,
        poseCurrentPromise,
        sceneGeometryPromise,
        livePreviewPromise,
        timeSyncPromise,
      ]);
      this._sensing = latest && typeof latest === "object" ? latest : null;
      this._pose = poseCurrent && typeof poseCurrent === "object" ? poseCurrent : null;
      this._sceneGeometry = sceneGeometry;
      this._livePreview = livePreview && typeof livePreview === "object" ? livePreview : null;
      this._timeSync = timeSync && typeof timeSync === "object" ? timeSync : null;
      if (this._sensing || this._sceneGeometry || this._livePreview || this._timeSync) {
        this._markActive();
      }
    } catch {}
  }

  _stabilizePersons(persons) {
    const next = Array.isArray(persons) ? persons.map((person) => ({ ...person })) : [];
    const now = Date.now();
    let currentWearerIndex = next.findIndex((person) => Boolean(person?.isWearer));
    const rememberedWearerFresh = Boolean(this._lastConfirmedWearerTrackId)
      && now - this._lastConfirmedWearerTrackIdAt <= 2_000;
    if (currentWearerIndex < 0 && rememberedWearerFresh) {
      const rememberedIndex = next.findIndex(
        (person) => String(person?.id || "") === String(this._lastConfirmedWearerTrackId || "")
      );
      if (rememberedIndex >= 0) {
        next[rememberedIndex] = {
          ...next[rememberedIndex],
          isWearer: true,
          source: `${next[rememberedIndex]?.source || "person"}:wearer_track_hold`,
          confidence: Math.max(numberOr(next[rememberedIndex]?.confidence, 0), 0.6),
        };
        currentWearerIndex = rememberedIndex;
      }
    }
    const currentWearer = currentWearerIndex >= 0 ? next[currentWearerIndex] : null;
    const currentWearerHasRaw =
      Array.isArray(currentWearer?.raw_keypoints) && currentWearer.raw_keypoints.length >= 17;

    if (currentWearer?.id) {
      this._lastConfirmedWearerTrackId = String(currentWearer.id);
      this._lastConfirmedWearerTrackIdAt = now;
    }

    if (currentWearerHasRaw) {
      this._lastWearerPerson = cloneJson(currentWearer);
      this._lastWearerPersonAt = now;
      return next;
    }

    const cacheFresh =
      this._lastWearerPerson && now - this._lastWearerPersonAt <= 2_000;
    if (!cacheFresh) {
      return next;
    }

    const cachedWearer = {
      ...cloneJson(this._lastWearerPerson),
      isWearer: true,
      source: `${this._lastWearerPerson?.source || "wearer"}:hold`,
    };
    const sameIdIndex = next.findIndex(
      (person) => String(person?.id || "") === String(cachedWearer?.id || "")
    );
    if (currentWearerIndex >= 0) {
      next[currentWearerIndex] = {
        ...currentWearer,
        ...cachedWearer,
        id: currentWearer?.id || cachedWearer.id,
        confidence: Math.max(
          numberOr(currentWearer?.confidence, 0),
          numberOr(cachedWearer?.confidence, 0)
        ),
      };
      return next;
    }

    if (sameIdIndex >= 0) {
      next[sameIdIndex] = {
        ...next[sameIdIndex],
        isWearer: true,
        source: `${next[sameIdIndex]?.source || "person"}:wearer_hold`,
        confidence: Math.max(
          numberOr(next[sameIdIndex]?.confidence, 0),
          numberOr(cachedWearer?.confidence, 0)
        ),
      };
      return next;
    }

    // Avoid injecting a stale ghost wearer when live data still has visible people
    // but phone->stereo matching has temporarily dropped.
    if (next.length > 0) {
      return next;
    }

    return [cachedWearer].slice(0, 1);
  }

  _deriveLivePhonePose(persons) {
    const now = Date.now();
    const wearer =
      (Array.isArray(persons)
        ? persons.find((person) => Boolean(person?.isWearer))
        : null) || null;
    const estimated = estimateWearerPhonePose(wearer, this._lastLivePhonePose);
    const edgePhonePose = normalizeEdgePhonePose(
      this._fusion?.operator_debug?.iphone_capture || null,
      this._sceneGeometry
    );
    const mergedEdgePhonePose = this._mergeLivePhonePose(edgePhonePose, estimated);
    if (mergedEdgePhonePose) {
      this._lastLivePhonePose = mergedEdgePhonePose;
      this._lastLivePhonePoseAt = now;
      return mergedEdgePhonePose;
    }
    if (estimated) {
      this._lastLivePhonePose = estimated;
      this._lastLivePhonePoseAt = now;
      return estimated;
    }
    if (this._lastLivePhonePose && now - this._lastLivePhonePoseAt <= 2_000) {
      return {
        ...cloneJson(this._lastLivePhonePose),
        source: `${this._lastLivePhonePose.source || "wearer_body_estimate"}:hold`,
      };
    }
    return null;
  }

  _mergeLivePhonePose(edgePose, estimatedPose) {
    if (!edgePose) {
      return estimatedPose || null;
    }
    const targetSpace = String(edgePose.target_space || "").trim();
    if (targetSpace !== "ios_arkit_world_frame") {
      return mergeEdgePhonePoseWithEstimate(edgePose, estimatedPose);
    }
    const aligned = this._alignArkitPhonePoseToRoom(edgePose, estimatedPose);
    return mergeEdgePhonePoseWithEstimate(aligned, estimatedPose);
  }

  _alignArkitPhonePoseToRoom(edgePose, estimatedPose) {
    const position = Array.isArray(edgePose?.position_m) ? edgePose.position_m.map((value) => numberOr(value, 0)) : null;
    const arkitBasis = buildBasisMatrix(edgePose?.right_vector, edgePose?.up_vector, edgePose?.forward_vector);
    const roomBasis = buildBasisMatrix(
      estimatedPose?.right_vector,
      estimatedPose?.up_vector,
      estimatedPose?.forward_vector
    );
    if (position && arkitBasis && roomBasis && Array.isArray(estimatedPose?.position_m)) {
      this._arkitPhoneRoomAnchor = {
        arkitOrigin: position,
        roomOrigin: estimatedPose.position_m.map((value) => numberOr(value, 0)),
        transform: multiplyMat3(roomBasis, transpose3(arkitBasis)),
      };
    }
    const anchor = this._arkitPhoneRoomAnchor;
    if (!anchor || !position) {
      return estimatedPose || null;
    }
    const delta = [
      position[0] - anchor.arkitOrigin[0],
      position[1] - anchor.arkitOrigin[1],
      position[2] - anchor.arkitOrigin[2],
    ];
    const roomDelta = multiplyMat3Vec3(anchor.transform, delta);
    const roomPosition = [
      anchor.roomOrigin[0] + roomDelta[0],
      anchor.roomOrigin[1] + roomDelta[1],
      anchor.roomOrigin[2] + roomDelta[2],
    ];
    const rotateVector = (vector) => (
      Array.isArray(vector) && vector.length >= 3
        ? multiplyMat3Vec3(anchor.transform, vector.map((value) => numberOr(value, 0)))
        : vector
    );
    return {
      ...cloneJson(edgePose),
      position_m: roomPosition,
      right_vector: rotateVector(edgePose.right_vector),
      up_vector: rotateVector(edgePose.up_vector),
      forward_vector: rotateVector(edgePose.forward_vector),
      target_space: "room_world_frame",
      source: `${edgePose.source || "ios_arkit_world_transform"}+arkit_room_aligned`,
    };
  }

  getSnapshot() {
    const edgeOverlay = buildEdgeOverlay(this._fusion, this._teleop);
    const hasEdgeContext = Boolean(
      this._sceneGeometry
      || this._livePreview
      || this._timeSync
      || edgeOverlay?.association
      || edgeOverlay?.fusion
      || edgeOverlay?.iphone
      || edgeOverlay?.stereo
      || edgeOverlay?.teleop
    );
    if (!this._sensing && !hasEdgeContext) {
      return null;
    }

    const derivedPersons = this._sensing
      ? deriveChekPersons(
        this._pose,
        this._sensing,
        this._fusion,
        this._teleop,
        this._sceneGeometry,
        (this._lastConfirmedWearerTrackIdAt > 0
          && Date.now() - this._lastConfirmedWearerTrackIdAt <= 2_000)
          ? this._lastConfirmedWearerTrackId
          : "",
        this._livePreview
      )
      : [];
    const persons = this._stabilizePersons(derivedPersons);
    const livePhonePose = this._deriveLivePhonePose(persons);
    const modelLoaded = Boolean(this._pose?.model_status?.loaded || this._sensing?.model_status?.loaded);
    const classification = {
      ...((this._sensing?.classification) || {}),
      presence: modelLoaded
        ? persons.length > 0
        : Boolean(this._sensing?.classification?.presence) || persons.length > 0,
      motion_level: modelLoaded
        ? (persons.length > 0 ? this._sensing?.classification?.motion_level || "present_still" : "absent")
        : this._sensing?.classification?.motion_level || (persons.length > 0 ? "present_still" : "absent")
    };
    return {
      ...(this._sensing || {}),
      classification,
      vital_signs: normalizeVitals(this._sensing?.vital_signs),
      persons,
      estimated_persons: modelLoaded ? persons.length : (this._sensing?.estimated_persons ?? persons.length),
      edge: edgeOverlay,
      scene_geometry: this._sceneGeometry,
      live_preview: this._livePreview,
      time_sync: this._timeSync,
      live_phone_pose: livePhonePose,
      stereo_preview_url: this._config.stereoPreviewUrl
    };
  }
}
