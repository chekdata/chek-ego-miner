/**
 * RuView Observatory — Main Scene Orchestrator
 *
 * Room-based WiFi sensing visualization with:
 * - Pool of 4 human wireframe figures (multi-person scenarios)
 * - 7 pose types (standing, walking, lying, sitting, fallen, exercising, gesturing, crouching)
 * - Scenario-specific room props (chair, exercise mat, door, rubble wall, screen, desk)
 * - Dot-matrix mist body mass, particle trails, WiFi waves, signal field
 * - Reflective floor, settings dialog, and practical data HUD
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { DemoDataGenerator } from './demo-data.js?v=20260317-geometry-markers-1';
import { NebulaBackground } from './nebula-background.js?v=20260317-geometry-markers-1';
import { PostProcessing } from './post-processing.js?v=20260317-geometry-markers-1';
import { ChekLiveAdapter, readChekConfigFromLocation } from './chek-live-adapter.js?v=20260318-motion-state-1';
import { FigurePool, SKELETON_PAIRS } from './figure-pool.js?v=20260317-stereo-plausibility-1';
import { PoseSystem } from './pose-system.js?v=20260317-geometry-markers-1';
import { ScenarioProps } from './scenario-props.js?v=20260317-geometry-markers-1';
import { HudController, DEFAULTS, SETTINGS_VERSION, PRESETS, SCENARIO_NAMES, migrateSettings } from './hud-controller.js?v=20260318-motion-state-1';

// ---- Palette ----
const C = {
  brandBlue:   0x3388ff,
  brandPink:   0xd377cd,
  brandGreen:  0x46ebd5,
  greenGlow:   0x46ebd5,
  greenBright: 0x8ff8ec,
  greenDim:    0x143742,
  amber:       0xd377cd,
  blueSignal:  0x3388ff,
  redAlert:    0xff628e,
  redHeart:    0xd377cd,
  bgDeep:      0x02050c,
  bgMid:       0x0a1220,
  whiteSoft:   0xeaf3ff,
};

function selectPrimaryObservatoryPerson(persons) {
  if (!Array.isArray(persons) || persons.length === 0) {
    return null;
  }
  return persons.find((person) => person?.isWearer) || persons[0] || null;
}

function createMarkerLabel(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return new THREE.Sprite();
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(3, 8, 20, 0.72)';
  ctx.strokeStyle = `#${color.toString(16).padStart(6, '0')}`;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(8, 8, canvas.width - 16, canvas.height - 16, 18);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#eaf3ff';
  ctx.font = '600 34px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
    })
  );
  sprite.scale.set(0.8, 0.3, 1);
  return sprite;
}

function applyBasisRotation(group, basis) {
  if (!group || !basis) {
    return false;
  }
  const right = Array.isArray(basis.right_vector) ? basis.right_vector : null;
  const up = Array.isArray(basis.up_vector) ? basis.up_vector : null;
  const forward = Array.isArray(basis.forward_vector) ? basis.forward_vector : null;
  if (!right || !up || !forward) {
    return false;
  }
  const rightVec = new THREE.Vector3(right[0], right[1], right[2]);
  const upVec = new THREE.Vector3(up[0], up[1], up[2]);
  const forwardVec = new THREE.Vector3(forward[0], forward[1], forward[2]);
  if (rightVec.lengthSq() < 1e-6 || upVec.lengthSq() < 1e-6 || forwardVec.lengthSq() < 1e-6) {
    return false;
  }
  rightVec.normalize();
  forwardVec.normalize();
  upVec.copy(new THREE.Vector3().crossVectors(forwardVec, rightVec)).normalize();
  rightVec.copy(new THREE.Vector3().crossVectors(upVec, forwardVec)).normalize();
  const basisMatrix = new THREE.Matrix4().makeBasis(rightVec, upVec, forwardVec);
  group.quaternion.setFromRotationMatrix(basisMatrix);
  return true;
}

// SCENARIO_NAMES, DEFAULTS, SETTINGS_VERSION, PRESETS imported from hud-controller.js

// ---- Main Class ----

class Observatory {
  constructor() {
    this._canvas = document.getElementById('observatory-canvas');
    this.settings = { ...DEFAULTS };
    this._chekConfig = readChekConfigFromLocation();
    this._liveMode = Boolean(this._chekConfig.enabled);

    // Load saved settings
    try {
      const ver = localStorage.getItem('ruview-settings-version');
      const saved = localStorage.getItem('ruview-observatory-settings');
      if (saved) {
        Object.assign(this.settings, migrateSettings(JSON.parse(saved), ver));
        if (ver !== SETTINGS_VERSION) {
          localStorage.setItem('ruview-observatory-settings', JSON.stringify(this.settings));
        }
      }
      if (ver !== SETTINGS_VERSION) {
        localStorage.setItem('ruview-settings-version', SETTINGS_VERSION);
      }
    } catch {}

    // Renderer
    this._renderer = new THREE.WebGLRenderer({
      canvas: this._canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, this._liveMode ? 1.25 : 2));
    this._renderer.setSize(window.innerWidth, window.innerHeight);
    this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this._renderer.toneMappingExposure = this.settings.exposure;
    this._renderer.shadowMap.enabled = true;
    this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Scene
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(C.bgDeep);
    this._scene.fog = new THREE.FogExp2(C.bgDeep, 0.005);

    // Camera
    this._camera = new THREE.PerspectiveCamera(
      this.settings.fov, window.innerWidth / window.innerHeight, 0.1, 300
    );
    this._camera.position.set(6, 5, 8);
    this._camera.lookAt(0, 1.2, 0);

    // Controls
    this._controls = new OrbitControls(this._camera, this._canvas);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.08;
    this._controls.minDistance = 2;
    this._controls.maxDistance = 25;
    this._controls.maxPolarAngle = Math.PI * 0.88;
    this._controls.target.set(0, 1.2, 0);
    this._controls.update();

    this._clock = new THREE.Clock();
    this._renderDirect = this._liveMode;

    // Data
    this._demoData = new DemoDataGenerator();
    this._demoData.setCycleDuration(this.settings.cycle || 30);
    if (this.settings.scenario && this.settings.scenario !== 'auto') {
      this._demoData.setScenario(this.settings.scenario);
    }
    this._currentData = null;
    this._currentScenario = null;
    this._liveAdapter = null;
    this._handleChekSnapshotMessage = (event) => {
      if (!this._liveAdapter || event.origin !== window.location.origin) {
        return;
      }
      const payload = event.data;
      if (!payload || payload.type !== 'chek-observatory-snapshot') {
        return;
      }
      this._liveAdapter.applyExternalSnapshot(payload.payload || null);
    };
    window.addEventListener('message', this._handleChekSnapshotMessage);

    // Build scene
    this._setupLighting();
    this._nebula = new NebulaBackground(this._scene);
    this._buildRoom();
    this._buildRouter();
    this._buildSceneGeometryMarkers();
    this._poseSystem = new PoseSystem();
    this._figurePool = new FigurePool(this._scene, this.settings, this._poseSystem);
    this._scenarioProps = new ScenarioProps(this._scene);
    this._buildDotMatrixMist();
    this._buildParticleTrail();
    this._buildWifiWaves();
    this._buildSignalField();
    this._applyColors();

    // Post-processing
    this._postProcessing = new PostProcessing(this._renderer, this._scene, this._camera);
    if (this._renderDirect) {
      this._postProcessing.setQuality(0);
    }
    this._applyPostSettings();

    // HUD controller (settings dialog, sparkline, vital displays)
    this._hud = new HudController(this);

    // State
    this._autopilot = false;
    this._autoAngle = 0;
    this._fpsFrames = 0;
    this._fpsTime = 0;
    this._fpsValue = 60;
    this._showFps = false;
    this._qualityLevel = 2;

    // WebSocket for live data — always try auto-detect on startup
    this._ws = null;
    this._liveData = null;
    this._autoDetectLive();

    // Input
    this._initKeyboard();
    this._hud.initSettings();
    this._hud.initQuickSelect();
    this._hud.applyMode({ liveMode: this._liveMode });
    window.addEventListener('resize', () => this._onResize());

    // Start
    this._animate();
  }

  // ---- Lighting ----

  _setupLighting() {
    this._ambient = new THREE.AmbientLight(C.whiteSoft, this.settings.ambient * 5.0);
    this._scene.add(this._ambient);

    const hemi = new THREE.HemisphereLight(C.brandBlue, C.bgMid, 1.25);
    this._scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(4, 8, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 20;
    key.shadow.camera.left = -8;
    key.shadow.camera.right = 8;
    key.shadow.camera.top = 8;
    key.shadow.camera.bottom = -8;
    this._scene.add(key);

    // Fill light from opposite side
    const fill = new THREE.DirectionalLight(C.brandGreen, 0.55);
    fill.position.set(-4, 5, -2);
    this._scene.add(fill);

    // Rim light from above/behind for edge definition
    const rim = new THREE.DirectionalLight(C.brandPink, 0.45);
    rim.position.set(0, 6, -5);
    this._scene.add(rim);

    // Overhead room light — general illumination
    const overhead = new THREE.PointLight(C.brandBlue, 1.15, 20, 1.0);
    overhead.position.set(0, 3.8, 0);
    this._scene.add(overhead);
  }

  // ---- Room ----

  _buildRoom() {
    this._grid = new THREE.GridHelper(12, 24, 0x1b4e90, 0x0a203e);
    this._grid.material.opacity = 0.5;
    this._grid.material.transparent = true;
    this._scene.add(this._grid);

    const boxGeo = new THREE.BoxGeometry(12, 4, 10);
    const edges = new THREE.EdgesGeometry(boxGeo);
    this._roomWire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
      color: C.blueSignal, opacity: 0.24, transparent: true,
    }));
    this._roomWire.position.y = 2;
    this._scene.add(this._roomWire);

    // Reflective floor
    const floorGeo = new THREE.PlaneGeometry(12, 10);
    this._floorMat = new THREE.MeshStandardMaterial({
      color: 0x060b14,
      roughness: 1.0 - this.settings.reflect * 0.7,
      metalness: this.settings.reflect * 0.5,
      emissive: 0x091223,
      emissiveIntensity: 0.15,
    });
    const floor = new THREE.Mesh(floorGeo, this._floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this._scene.add(floor);

    // Table under router
    const tableGeo = new THREE.BoxGeometry(0.8, 0.6, 0.5);
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x1a2233, roughness: 0.55, emissive: 0x101828, emissiveIntensity: 0.28 });
    const table = new THREE.Mesh(tableGeo, tableMat);
    table.position.set(-4, 0.3, -3);
    table.castShadow = true;
    this._scene.add(table);
  }

  // ---- Router ----

  _buildRouter() {
    this._routerGroup = new THREE.Group();
    this._routerGroup.position.set(-4, 0.92, -3);
    this._routerDefaultPosition = this._routerGroup.position.clone();
    this._routerDefaultRotation = this._routerGroup.rotation.clone();

    const bodyGeo = new THREE.BoxGeometry(0.6, 0.12, 0.35);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2b3347, roughness: 0.2, metalness: 0.7, emissive: 0x161d31, emissiveIntensity: 0.24 });
    this._routerGroup.add(new THREE.Mesh(bodyGeo, bodyMat));

    for (let i = -1; i <= 1; i++) {
      const antGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.35);
      const antMat = new THREE.MeshStandardMaterial({ color: 0x5b6881, roughness: 0.3, metalness: 0.6, emissive: 0x161d2d, emissiveIntensity: 0.18 });
      const ant = new THREE.Mesh(antGeo, antMat);
      ant.position.set(i * 0.2, 0.24, 0);
      ant.rotation.z = i * 0.15;
      this._routerGroup.add(ant);
    }

    const ledGeo = new THREE.SphereGeometry(0.025);
    this._routerLed = new THREE.Mesh(ledGeo, new THREE.MeshBasicMaterial({ color: C.greenGlow }));
    this._routerLed.position.set(0.22, 0.07, 0.18);
    this._routerGroup.add(this._routerLed);

    this._routerLight = new THREE.PointLight(C.blueSignal, 1.2, 8);
    this._routerLight.position.set(0, 0.3, 0);
    this._routerGroup.add(this._routerLight);

    this._scene.add(this._routerGroup);
  }

  _buildSceneGeometryMarkers() {
    this._sceneGeometryMarkersGroup = new THREE.Group();
    this._scene.add(this._sceneGeometryMarkersGroup);

    this._apMarkerGroups = [];
    this._stereoRigGroup = new THREE.Group();
    this._stereoRigGroup.visible = false;

    const stereoBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.08, 0.08),
      new THREE.MeshStandardMaterial({
        color: C.brandBlue,
        roughness: 0.28,
        metalness: 0.55,
        emissive: C.brandBlue,
        emissiveIntensity: 0.14,
      })
    );
    this._stereoRigGroup.add(stereoBody);

    const stereoLensLeft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.022, 0.022, 0.02, 18),
      new THREE.MeshBasicMaterial({ color: C.brandGreen })
    );
    stereoLensLeft.rotation.z = Math.PI / 2;
    stereoLensLeft.position.set(-0.06, 0, 0.05);
    this._stereoRigGroup.add(stereoLensLeft);

    const stereoLensRight = stereoLensLeft.clone();
    stereoLensRight.position.x = 0.06;
    this._stereoRigGroup.add(stereoLensRight);

    const stereoArrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.035, 0.12, 12),
      new THREE.MeshBasicMaterial({ color: C.brandBlue })
    );
    stereoArrow.rotation.x = -Math.PI / 2;
    stereoArrow.position.z = 0.14;
    this._stereoRigGroup.add(stereoArrow);

    const stereoLabel = createMarkerLabel('双目', C.brandBlue);
    stereoLabel.position.set(0, 0.22, 0);
    this._stereoRigGroup.add(stereoLabel);
    this._sceneGeometryMarkersGroup.add(this._stereoRigGroup);

    this._phonePoseGroup = new THREE.Group();
    this._phonePoseGroup.visible = false;

    const phoneBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.085, 0.16, 0.012),
      new THREE.MeshStandardMaterial({
        color: 0x0d1528,
        roughness: 0.22,
        metalness: 0.65,
        emissive: C.brandGreen,
        emissiveIntensity: 0.18,
      })
    );
    this._phonePoseGroup.add(phoneBody);

    const phoneScreen = new THREE.Mesh(
      new THREE.BoxGeometry(0.068, 0.128, 0.003),
      new THREE.MeshBasicMaterial({
        color: C.brandGreen,
        transparent: true,
        opacity: 0.55,
      })
    );
    phoneScreen.position.z = 0.0085;
    this._phonePoseGroup.add(phoneScreen);

    const phoneArrow = new THREE.Group();
    const arrowShaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.008, 0.008, 0.22),
      new THREE.MeshBasicMaterial({ color: C.brandPink })
    );
    arrowShaft.rotation.x = Math.PI / 2;
    arrowShaft.position.z = 0.12;
    phoneArrow.add(arrowShaft);

    const arrowHead = new THREE.Mesh(
      new THREE.ConeGeometry(0.025, 0.06, 12),
      new THREE.MeshBasicMaterial({ color: C.brandPink })
    );
    arrowHead.rotation.x = -Math.PI / 2;
    arrowHead.position.z = 0.25;
    phoneArrow.add(arrowHead);
    this._phonePoseGroup.add(phoneArrow);

    const phoneAura = new THREE.Mesh(
      new THREE.RingGeometry(0.08, 0.1, 32),
      new THREE.MeshBasicMaterial({
        color: C.brandGreen,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      })
    );
    phoneAura.rotation.x = -Math.PI / 2;
    phoneAura.position.y = -0.085;
    this._phonePoseGroup.add(phoneAura);

    const phoneLabel = createMarkerLabel('手机', C.brandPink);
    phoneLabel.position.set(0, 0.2, 0);
    this._phonePoseGroup.add(phoneLabel);

    this._sceneGeometryMarkersGroup.add(this._phonePoseGroup);
    this._applySceneGeometryMarkerVisibility();
  }

  _createApMarker(index) {
    const group = new THREE.Group();
    group.visible = false;

    const beacon = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.05, 0.28, 16),
      new THREE.MeshStandardMaterial({
        color: C.brandBlue,
        roughness: 0.3,
        metalness: 0.5,
        emissive: C.brandBlue,
        emissiveIntensity: 0.12,
      })
    );
    beacon.position.y = 0.14;
    group.add(beacon);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.12, 0.16, 24),
      new THREE.MeshBasicMaterial({
        color: C.brandGreen,
        transparent: true,
        opacity: 0.38,
        side: THREE.DoubleSide,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    group.add(ring);

    const label = createMarkerLabel(`AP ${index + 1}`, C.brandGreen);
    label.position.set(0, 0.42, 0);
    group.add(label);
    this._sceneGeometryMarkersGroup.add(group);
    this._apMarkerGroups.push(group);
    return group;
  }

  _applySceneGeometryMarkerVisibility() {
    const visible = Boolean(this.settings.geometryMarkers);
    if (this._sceneGeometryMarkersGroup) {
      this._sceneGeometryMarkersGroup.visible = visible;
    }
  }

  _updateSceneGeometryMarkers(data) {
    const sceneGeometry = data?.scene_geometry?.geometry || null;
    const apNodes = Array.isArray(sceneGeometry?.ap_nodes) ? sceneGeometry.ap_nodes : [];
    const apAnchor = apNodes.length > 0 ? apNodes[0] : null;
    const stereoRig = sceneGeometry?.stereo_rig || null;
    const phonePose = data?.live_phone_pose || sceneGeometry?.phone_pose || null;

    if (apAnchor?.position_m?.length === 3) {
      this._routerGroup.position.set(apAnchor.position_m[0], apAnchor.position_m[1], apAnchor.position_m[2]);
      if (Array.isArray(apAnchor.rotation_deg) && apAnchor.rotation_deg.length === 3) {
        this._routerGroup.rotation.set(
          THREE.MathUtils.degToRad(apAnchor.rotation_deg[0] || 0),
          THREE.MathUtils.degToRad(apAnchor.rotation_deg[1] || 0),
          THREE.MathUtils.degToRad(apAnchor.rotation_deg[2] || 0)
        );
      }
    } else {
      this._routerGroup.position.copy(this._routerDefaultPosition);
      this._routerGroup.rotation.copy(this._routerDefaultRotation);
    }

    for (const wave of this._wifiWaves || []) {
      wave.mesh.position.copy(this._routerGroup.position);
      wave.mesh.position.y += 0.5;
    }

    apNodes.forEach((apNode, index) => {
      const marker = this._apMarkerGroups[index] || this._createApMarker(index);
      if (apNode?.position_m?.length === 3) {
        marker.visible = true;
        marker.position.set(apNode.position_m[0], apNode.position_m[1], apNode.position_m[2]);
        if (Array.isArray(apNode.rotation_deg) && apNode.rotation_deg.length === 3) {
          marker.rotation.set(
            THREE.MathUtils.degToRad(apNode.rotation_deg[0] || 0),
            THREE.MathUtils.degToRad(apNode.rotation_deg[1] || 0),
            THREE.MathUtils.degToRad(apNode.rotation_deg[2] || 0)
          );
        } else {
          marker.rotation.set(0, 0, 0);
        }
      } else {
        marker.visible = false;
      }
    });
    for (let index = apNodes.length; index < this._apMarkerGroups.length; index += 1) {
      this._apMarkerGroups[index].visible = false;
    }

    if (stereoRig?.position_m?.length === 3) {
      this._stereoRigGroup.visible = true;
      this._stereoRigGroup.position.set(stereoRig.position_m[0], stereoRig.position_m[1], stereoRig.position_m[2]);
      if (Array.isArray(stereoRig.rotation_deg) && stereoRig.rotation_deg.length === 3) {
        this._stereoRigGroup.rotation.set(
          THREE.MathUtils.degToRad(stereoRig.rotation_deg[0] || 0),
          THREE.MathUtils.degToRad(stereoRig.rotation_deg[1] || 0),
          THREE.MathUtils.degToRad(stereoRig.rotation_deg[2] || 0)
        );
      } else {
        this._stereoRigGroup.rotation.set(0, 0, 0);
      }
    } else {
      this._stereoRigGroup.visible = false;
    }

    if (phonePose?.position_m?.length === 3) {
      this._phonePoseGroup.visible = true;
      this._phonePoseGroup.position.set(phonePose.position_m[0], phonePose.position_m[1], phonePose.position_m[2]);
      if (!applyBasisRotation(this._phonePoseGroup, phonePose)
          && Array.isArray(phonePose.rotation_deg) && phonePose.rotation_deg.length === 3) {
        this._phonePoseGroup.rotation.set(
          THREE.MathUtils.degToRad(phonePose.rotation_deg[0] || 0),
          THREE.MathUtils.degToRad(phonePose.rotation_deg[1] || 0),
          THREE.MathUtils.degToRad(phonePose.rotation_deg[2] || 0)
        );
      } else if (!phonePose.forward_vector) {
        this._phonePoseGroup.rotation.set(0, 0, 0);
      }
    } else {
      this._phonePoseGroup.visible = false;
    }
    this._applySceneGeometryMarkerVisibility();
  }

  // ---- WiFi Waves ----

  _buildWifiWaves() {
    this._wifiWaves = [];
    for (let i = 0; i < 5; i++) {
      const radius = 0.8 + i * 1.0;
      const geo = new THREE.SphereGeometry(radius, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.6);
      const mat = new THREE.MeshBasicMaterial({
        color: C.blueSignal,
        transparent: true, opacity: 0,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false, wireframe: true,
      });
      const shell = new THREE.Mesh(geo, mat);
      shell.position.copy(this._routerGroup.position);
      shell.position.y += 0.5;
      this._scene.add(shell);
      this._wifiWaves.push({ mesh: shell, mat, phase: i * 0.7 });
    }
  }

  // ========================================
  // DOT MATRIX MIST
  // ========================================

  _buildDotMatrixMist() {
    const COUNT = 800;
    const positions = new Float32Array(COUNT * 3);
    const alphas = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 0.5;
      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = Math.random() * 1.8;
      positions[i * 3 + 2] = Math.sin(angle) * r;
      alphas[i] = 0;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float alpha;
        varying float vAlpha;
        void main() {
          vAlpha = alpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = 3.0 * (200.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vAlpha;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float edge = smoothstep(0.5, 0.2, d);
          gl_FragColor = vec4(uColor, edge * vAlpha);
        }
      `,
      uniforms: { uColor: { value: new THREE.Color(this.settings.wireColor) } },
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this._mistPoints = new THREE.Points(geo, mat);
    this._scene.add(this._mistPoints);
    this._mistCount = COUNT;
  }

  // ---- Particle Trail ----

  _buildParticleTrail() {
    const COUNT = 200;
    const positions = new Float32Array(COUNT * 3);
    const ages = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) ages[i] = 1;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('age', new THREE.BufferAttribute(ages, 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float age;
        varying float vAge;
        void main() {
          vAge = age;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = max(1.0, (1.0 - age) * 5.0 * (150.0 / -mv.z));
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vAge;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float alpha = (1.0 - vAge) * 0.6 * smoothstep(0.5, 0.1, d);
          gl_FragColor = vec4(uColor, alpha);
        }
      `,
      uniforms: { uColor: { value: new THREE.Color(C.brandPink) } },
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    this._trail = new THREE.Points(geo, mat);
    this._scene.add(this._trail);
    this._trailHead = 0;
    this._trailCount = COUNT;
    this._trailTimer = 0;
  }

  // ---- Signal Field ----

  _buildSignalField() {
    const gridSize = 20;
    const count = gridSize * gridSize;
    const positions = new Float32Array(count * 3);
    this._fieldColors = new Float32Array(count * 3);
    this._fieldSizes = new Float32Array(count);
    for (let iz = 0; iz < gridSize; iz++) {
      for (let ix = 0; ix < gridSize; ix++) {
        const idx = iz * gridSize + ix;
        positions[idx * 3] = (ix - gridSize / 2) * 0.6;
        positions[idx * 3 + 1] = 0.02;
        positions[idx * 3 + 2] = (iz - gridSize / 2) * 0.5;
        this._fieldSizes[idx] = 8;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this._fieldColors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(this._fieldSizes, 1));
    this._fieldMat = new THREE.PointsMaterial({
      size: 0.35, vertexColors: true, transparent: true,
      opacity: this.settings.field, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true,
    });
    this._fieldPoints = new THREE.Points(geo, this._fieldMat);
    this._scene.add(this._fieldPoints);
  }

  // ---- Keyboard ----

  _initKeyboard() {
    window.addEventListener('keydown', (e) => {
      if (this._hud.settingsOpen) return;
      switch (e.key.toLowerCase()) {
        case 'a':
          this._autopilot = !this._autopilot;
          this._controls.enabled = !this._autopilot;
          break;
        case 'd':
          if (!this._liveMode) {
            this._demoData.cycleScenario();
          }
          break;
        case 'f':
          this._showFps = !this._showFps;
          document.getElementById('fps-counter').style.display = this._showFps ? 'block' : 'none';
          break;
        case 's': this._hud.toggleSettings(); break;
        case ' ':
          if (!this._liveMode) {
            e.preventDefault();
            this._demoData.paused = !this._demoData.paused;
          }
          break;
      }
    });
  }

  // ---- Settings / HUD methods delegated to HudController ----

  _applyPostSettings() {
    const pp = this._postProcessing;
    pp._bloomPass.strength = this.settings.bloom;
    pp._bloomPass.radius = this.settings.bloomRadius;
    pp._bloomPass.threshold = this.settings.bloomThresh;
    pp._vignettePass.uniforms.uVignetteStrength.value = this.settings.vignette;
    pp._vignettePass.uniforms.uGrainStrength.value = this.settings.grain;
    pp._vignettePass.uniforms.uChromaticStrength.value = this.settings.chromatic;
  }

  _applyColors() {
    const wc = new THREE.Color(this.settings.wireColor);
    const jc = new THREE.Color(this.settings.jointColor);
    const trailColor = wc.clone().lerp(jc, 0.35);
    this._figurePool.applyColors(wc, jc);
    this._mistPoints.material.uniforms.uColor.value.copy(wc);
    this._trail.material.uniforms.uColor.value.copy(trailColor);
  }

  // ---- WebSocket live data ----

  _autoDetectLive() {
    if (this._chekConfig.enabled) {
      this._liveAdapter = new ChekLiveAdapter(this._chekConfig);
      this._liveAdapter.start();
      this.settings.dataSource = 'ws';
      this.settings.wsUrl = this._chekConfig.sensingWsUrl || this.settings.wsUrl;
      this._hud.updateSourceBadge('ws', true);
      console.log('[Observatory] CHEK live adapter enabled', this._chekConfig);
      return;
    }

    // Probe sensing server health on same origin, then common ports
    const host = window.location.hostname || 'localhost';
    const candidates = [
      window.location.origin,                   // same origin (e.g. :3000)
      `http://${host}:8765`,                     // default WS port
      `http://${host}:3000`,                     // default HTTP port
    ];
    // Deduplicate
    const unique = [...new Set(candidates)];

    const tryNext = (i) => {
      if (i >= unique.length) {
        console.log('[Observatory] No sensing server detected, using demo mode');
        return;
      }
      const base = unique[i];
      fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(data => {
          if (data && data.status === 'ok') {
            const wsProto = base.startsWith('https') ? 'wss:' : 'ws:';
            const urlObj = new URL(base);
            const wsUrl = `${wsProto}//${urlObj.host}/ws/sensing`;
            console.log('[Observatory] Sensing server detected at', base, '→', wsUrl);
            this.settings.dataSource = 'ws';
            this.settings.wsUrl = wsUrl;
            this._connectWS(wsUrl);
          } else {
            tryNext(i + 1);
          }
        })
        .catch(() => tryNext(i + 1));
    };
    tryNext(0);
  }

  _connectWS(url) {
    this._disconnectWS();
    try {
      this._ws = new WebSocket(url);
      this._ws.onopen = () => {
        console.log('[Observatory] WebSocket connected');
        this._hud.updateSourceBadge('ws', this._ws);
      };
      this._ws.onmessage = (evt) => { try { this._liveData = JSON.parse(evt.data); } catch {} };
      this._ws.onclose = () => {
        console.log('[Observatory] WebSocket closed, falling back to demo');
        this._ws = null;
        this.settings.dataSource = 'demo';
        this._hud.updateSourceBadge('demo', null);
      };
      this._ws.onerror = () => {};
    } catch {}
  }

  _disconnectWS() {
    if (this._ws) { this._ws.close(); this._ws = null; }
    this._liveData = null;
  }

  // ========================================
  // ANIMATION LOOP
  // ========================================

  _animate() {
    requestAnimationFrame(() => this._animate());
    const dt = Math.min(this._clock.getDelta(), 0.1);
    const elapsed = this._clock.getElapsedTime();

    // Data source
    const adapterFrame = this._liveAdapter ? this._liveAdapter.getSnapshot() : null;
    if (this._liveAdapter) {
      this._hud.updateSourceBadge('ws', this._liveAdapter.connected);
    }
    if (adapterFrame) {
      this._currentData = adapterFrame;
    } else if (this.settings.dataSource === 'ws' && this._liveData) {
      this._currentData = this._liveData;
    } else {
      this._currentData = this._demoData.update(dt);
    }
    const data = this._currentData;
    window.__CHEK_OBSERVATORY_SNAPSHOT__ = data;

    // Updates
    this._updateSceneGeometryMarkers(data);
    this._nebula.update(dt, elapsed);
    this._figurePool.update(data, elapsed);
    this._scenarioProps.update(data, this._demoData.currentScenario);
    this._updateDotMatrixMist(data, elapsed);
    this._updateParticleTrail(data, dt, elapsed);
    this._updateWifiWaves(elapsed);
    this._updateSignalField(data);
    this._hud.updateHUD(data, this._demoData);
    this._hud.updateSparkline(data);

    // Router LED
    this._routerLed.material.opacity = 0.5 + 0.5 * Math.sin(elapsed * 8);
    this._routerLight.intensity = 0.3 + 0.2 * Math.sin(elapsed * 3);

    // Autopilot orbit
    if (this._autopilot) {
      this._autoAngle += dt * this.settings.orbitSpeed;
      const r = 10;
      this._camera.position.set(
        Math.sin(this._autoAngle) * r,
        4.5 + Math.sin(this._autoAngle * 0.5),
        Math.cos(this._autoAngle) * r
      );
      this._controls.target.set(0, 1.2, 0);
      this._controls.update();
    }
    this._controls.update();
    if (this._renderDirect) {
      this._renderer.render(this._scene, this._camera);
    } else {
      this._postProcessing.update(elapsed);
      this._postProcessing.render();
    }
    this._updateFPS(dt);
  }


  // ========================================
  // MIST & TRAIL
  // ========================================

  _updateDotMatrixMist(data, elapsed) {
    const persons = data?.persons || [];
    const isPresent = data?.classification?.presence || false;
    const pos = this._mistPoints.geometry.attributes.position;
    const alpha = this._mistPoints.geometry.attributes.alpha;
    const primaryPerson = selectPrimaryObservatoryPerson(persons);

    if (!isPresent || !primaryPerson) {
      for (let i = 0; i < this._mistCount; i++) {
        alpha.array[i] = Math.max(0, alpha.array[i] - 0.02);
      }
      alpha.needsUpdate = true;
      return;
    }

    // Follow wearer first, then fallback to the current primary person.
    const pp = primaryPerson.position || [0, 0, 0];
    const rawKeypoints = Array.isArray(primaryPerson?.raw_keypoints) && primaryPerson.raw_keypoints.length >= 17
      ? primaryPerson.raw_keypoints
      : null;
    const px = pp[0] || 0, pz = pp[2] || 0;
    const ms = primaryPerson.motion_score || 0;
    const pose = primaryPerson.pose || 'standing';
    const isLying = pose === 'lying' || pose === 'fallen';
    const bodyH = isLying ? 0.4 : 1.7;
    const bodyBaseY = isLying ? (pp[1] || 0) + 0.05 : 0.05;
    const spread = ms > 50 ? 0.6 : 0.4;

    if (rawKeypoints) {
      for (let i = 0; i < this._mistCount; i++) {
        const joint = rawKeypoints[i % rawKeypoints.length];
        const jitter = 0.01 + (i % 5) * 0.002;
        const tx = joint[0] + Math.cos(elapsed * 0.7 + i * 0.37) * jitter;
        const ty = joint[1] + Math.sin(elapsed * 0.9 + i * 0.21) * jitter * 0.4;
        const tz = joint[2] + Math.sin(elapsed * 0.6 + i * 0.41) * jitter;

        pos.array[i * 3] += (tx - pos.array[i * 3]) * 0.08;
        pos.array[i * 3 + 1] += (ty - pos.array[i * 3 + 1]) * 0.08;
        pos.array[i * 3 + 2] += (tz - pos.array[i * 3 + 2]) * 0.08;

        const targetAlpha = 0.014 + Math.sin(elapsed * 2 + i * 0.5) * 0.006;
        alpha.array[i] += (targetAlpha - alpha.array[i]) * 0.08;
      }
      pos.needsUpdate = true;
      alpha.needsUpdate = true;
      return;
    }

    for (let i = 0; i < this._mistCount; i++) {
      const drift = Math.sin(elapsed * 0.5 + i * 0.1) * 0.003;
      const angle = (i / this._mistCount) * Math.PI * 2 + elapsed * 0.1;
      const layerT = (i % 20) / 20;
      const layerY = bodyBaseY + layerT * bodyH;

      let bodyWidth;
      if (isLying) {
        bodyWidth = 0.25;
      } else {
        bodyWidth = layerT > 0.75 ? 0.15 : (layerT > 0.45 ? 0.25 : 0.18);
      }
      const r = bodyWidth * (0.5 + 0.5 * Math.sin(i * 1.7 + elapsed * 0.3)) * spread;

      const tx = px + Math.cos(angle + i * 0.3) * r + drift;
      const tz = pz + Math.sin(angle + i * 0.5) * r * 0.6;

      pos.array[i * 3] += (tx - pos.array[i * 3]) * 0.05;
      pos.array[i * 3 + 1] += (layerY - pos.array[i * 3 + 1]) * 0.05;
      pos.array[i * 3 + 2] += (tz - pos.array[i * 3 + 2]) * 0.05;

      const targetAlpha = 0.15 + Math.sin(elapsed * 2 + i * 0.5) * 0.08;
      alpha.array[i] += (targetAlpha - alpha.array[i]) * 0.08;
    }
    pos.needsUpdate = true;
    alpha.needsUpdate = true;
  }

  _updateParticleTrail(data, dt, elapsed) {
    if (this.settings.trail <= 0) return;
    const persons = data?.persons || [];
    const isPresent = data?.classification?.presence || false;
    const pos = this._trail.geometry.attributes.position;
    const ages = this._trail.geometry.attributes.age;
    const primaryPerson = selectPrimaryObservatoryPerson(persons);

    for (let i = 0; i < this._trailCount; i++) {
      ages.array[i] = Math.min(1, ages.array[i] + dt * 0.8);
    }

    // Emit from all active persons
    if (isPresent && persons.length > 0) {
      this._trailTimer += dt;
      const ms = primaryPerson?.motion_score || 0;
      const emitRate = ms > 50 ? 0.02 : 0.08;

      if (this._trailTimer >= emitRate) {
        this._trailTimer = 0;
        for (const p of persons) {
          const pp = p.position || [0, 0, 0];
          const idx = this._trailHead;
          pos.array[idx * 3] = (pp[0] || 0) + (Math.random() - 0.5) * 0.15;
          pos.array[idx * 3 + 1] = Math.random() * 1.5 + 0.1;
          pos.array[idx * 3 + 2] = (pp[2] || 0) + (Math.random() - 0.5) * 0.15;
          ages.array[idx] = 0;
          this._trailHead = (this._trailHead + 1) % this._trailCount;
        }
      }
    }
    pos.needsUpdate = true;
    ages.needsUpdate = true;
  }

  // ---- WiFi Waves ----

  _updateWifiWaves(elapsed) {
    for (const w of this._wifiWaves) {
      const t = (elapsed * 0.8 + w.phase) % 4.5;
      const life = t / 4.5;
      w.mat.opacity = Math.max(0, this.settings.waves * 0.25 * (1 - life));
      const scale = 1 + life * 0.6;
      w.mesh.scale.set(scale, scale, scale);
      w.mesh.rotation.y = elapsed * 0.05;
    }
  }

  // ---- Signal Field ----

  _updateSignalField(data) {
    const field = data?.signal_field?.values;
    if (!field) return;
    const count = Math.min(field.length, 400);
    for (let i = 0; i < count; i++) {
      const v = field[i] || 0;
      let r;
      let g;
      let b;
      if (v < 0.5) {
        const t = v / 0.5;
        r = 0.2 + (0.2745 - 0.2) * t;
        g = 0.5333 + (0.9216 - 0.5333) * t;
        b = 1.0 + (0.8353 - 1.0) * t;
      } else {
        const t = (v - 0.5) / 0.5;
        r = 0.2745 + (0.8275 - 0.2745) * t;
        g = 0.9216 + (0.4667 - 0.9216) * t;
        b = 0.8353 + (0.8039 - 0.8353) * t;
      }
      this._fieldColors[i * 3] = r;
      this._fieldColors[i * 3 + 1] = g;
      this._fieldColors[i * 3 + 2] = b;
      this._fieldSizes[i] = 5 + v * 15;
    }
    this._fieldPoints.geometry.attributes.color.needsUpdate = true;
    this._fieldPoints.geometry.attributes.size.needsUpdate = true;
  }

  // ---- FPS ----

  _updateFPS(dt) {
    this._fpsFrames++;
    this._fpsTime += dt;
    if (this._fpsTime >= 1) {
      this._fpsValue = Math.round(this._fpsFrames / this._fpsTime);
      this._fpsFrames = 0;
      this._fpsTime = 0;
      if (this._showFps) {
        document.getElementById('fps-counter').textContent = `${this._fpsValue} FPS`;
      }
      this._adaptQuality();
    }
  }

  _adaptQuality() {
    let nl = this._qualityLevel;
    if (this._fpsValue < 25 && nl > 0) nl--;
    else if (this._fpsValue > 55 && nl < 2) nl++;
    if (nl !== this._qualityLevel) {
      this._qualityLevel = nl;
      this._nebula.setQuality(nl);
      this._postProcessing.setQuality(nl);
    }
  }

  _onResize() {
    const w = window.innerWidth, h = window.innerHeight;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
    this._postProcessing.resize(w, h);
  }
}

new Observatory();
