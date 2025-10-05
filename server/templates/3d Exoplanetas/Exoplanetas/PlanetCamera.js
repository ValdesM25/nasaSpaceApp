// Clase para manejar la cámara planetaria
class PlanetCamera {
    constructor(camera, renderer) {
        this.camera = camera;
        this.renderer = renderer;
        this.controls = null;
        this.currentMode = 'orbit'; // 'orbit' o 'fixed'
        this.targetObject = null;
        this.isActive = false;
        this.relativePosition = new THREE.Vector3();
        
        this.initControls();
    }
    
    initControls() {
        // Controles orbitales estándar
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 2;
        this.controls.maxDistance = 50;
    }
    
    // Activar cámara en un objeto específico
    activate(object, cameraPosition) {
        this.targetObject = object;
        this.isActive = true;
        
        // Posicionar cámara
        this.camera.position.copy(cameraPosition);
        
        // Calcular posición relativa inicial
        const objectPosition = new THREE.Vector3();
        object.getWorldPosition(objectPosition);
        this.relativePosition.copy(this.camera.position).sub(objectPosition);
        
        // Configurar controles para el objeto
        this.controls.target.copy(objectPosition);
        this.controls.update();
        
        // Configurar modo inicial
        this.setMode('orbit');
    }
    
    // Desactivar cámara
    deactivate() {
        this.isActive = false;
        this.targetObject = null;
        this.controls.reset();
    }
    
    // Cambiar modo de cámara
    setMode(mode) {
        this.currentMode = mode;
        
        if (mode === 'fixed') {
            // En modo fijo, deshabilitamos los controles para que la cámara no se mueva
            this.controls.enabled = false;
            
            // Recalcular posición relativa
            if (this.targetObject) {
                const objectPosition = new THREE.Vector3();
                this.targetObject.getWorldPosition(objectPosition);
                this.relativePosition.copy(this.camera.position).sub(objectPosition);
            }
        } else {
            // En modo orbital, habilitamos los controles normales
            this.controls.enabled = true;
        }
    }
    
    // Actualizar en cada frame (para modo fijo)
    update() {
        if (this.isActive && this.currentMode === 'fixed' && this.targetObject) {
            // En modo fijo, la cámara sigue al planeta manteniendo posición relativa
            const objectPosition = new THREE.Vector3();
            this.targetObject.getWorldPosition(objectPosition);
            
            // Mantener la cámara en posición relativa al planeta
            this.camera.position.copy(objectPosition).add(this.relativePosition);
            this.camera.lookAt(objectPosition);
        }
        
        if (this.controls.enabled) {
            this.controls.update();
        }
    }
    
    // Obtener modo actual
    getCurrentMode() {
        return this.currentMode;
    }
    
    // Verificar si está activa
    isCameraActive() {
        return this.isActive;
    }
}