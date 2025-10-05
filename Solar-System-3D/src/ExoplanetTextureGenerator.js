// ExoplanetTextureGenerator.js
// Genera texturas procedurales para exoplanetas a partir de parámetros físicos y atmosféricos.
// Uso:
//   import ExoplanetTextureGenerator from './ExoplanetTextureGenerator.js'
//   const gen = new ExoplanetTextureGenerator({ width: 1024, height: 512, starTint: 0.15 });
//   const tex = gen.generateTexture({
//     eqT: 300,
//     starRGB: { r: 1, g: 0.95, b: 0.9 },
//     planetaryRadius: 1.2,    // en radios terrestres
//     massEarth: 2.0,          // en masas terrestres
//     insolationFlux: 0.8,     // relativo a la Tierra
//     planetType: 'Auto',      // 'Auto' | 'Rocoso' | 'Gaseoso'
//     composition: { h2o: 0.3, co2: 0.2, ch4: 0.1, na: 0.0, silicates: 0.0 },
//     atmosphere: { pressureBars: 1.0, cloudCover: 0.4, haze: 0.1 }
//   });

import * as THREE from 'three';

export default class ExoplanetTextureGenerator {
  constructor(options = {}) {
    this.width = options.width ?? 1024;
    this.height = options.height ?? 512;
    this.starTint = options.starTint ?? 0.15; // fuerza del tinte por color de estrella
  }

  // API principal: devuelve THREE.CanvasTexture
  generateTexture(params) {
    const canvas = this.generateCanvas(params);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1, 1);
    tex.needsUpdate = true;
    return tex;
  }

  // API alternativa: devuelve el canvas por si quieres exportar como imagen
  generateCanvas(params) {
    const p = this.normalizeParams(params);
    const { eqT, starRGB, composition, atmosphere, planetType, massEarth, planetaryRadius, insolationFlux } = p;

    const c = document.createElement('canvas');
    c.width = this.width;
    c.height = this.height;
    const ctx = c.getContext('2d');

    // 1) Paleta base por T_eq
    let { base1, base2 } = this.#basePalette(eqT);

    // 2) Tinte por color estelar
    base1 = this.#tintHex(base1, starRGB, this.starTint);
    base2 = this.#tintHex(base2, starRGB, this.starTint);

    // 3) Sesgos de color por composición (H2O/CH4/CO2/Na/Silicatos)
    base1 = this.#shiftByComposition(base1, composition, eqT);
    base2 = this.#shiftByComposition(base2, composition, eqT);

    // 4) Fondo degradado
    const grad = ctx.createLinearGradient(0, 0, c.width, 0);
    grad.addColorStop(0, base1);
    grad.addColorStop(1, base2);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, c.width, c.height);

    // 5) Determinar tipo planetario si es 'Auto'
    let resolvedType = planetType;
    if (resolvedType === 'Auto') {
      const pr = Math.max(0.5, planetaryRadius);
      const densityLike = massEarth / Math.pow(pr, 3); // aproximado
      resolvedType = densityLike > 0.8 ? 'Rocoso' : 'Gaseoso';
    }

    // 6) Patrones principales
    if (resolvedType === 'Gaseoso') {
      // Bandas
      for (let i = 0; i < 90; i++) {
        const y = Math.floor((i + 1) * (c.height / 90));
        const alpha = 0.05 + 0.12 * Math.random();
        const dark = Math.random() < 0.5;
        ctx.fillStyle = dark ? `rgba(0,0,0,${alpha})` : `rgba(255,255,255,${alpha * 0.5})`;
        ctx.fillRect(0, y, c.width, 2 + Math.random() * 3);
      }
    } else {
      // Rocoso: mares y continentes
      const oceanBias = (composition.h2o > 0.3 && insolationFlux != null) ? this.#clamp01(insolationFlux) : 0;
      if (oceanBias > 0.2) {
        ctx.fillStyle = this.#tintHex('#1e6091', starRGB, 0.2);
        ctx.globalAlpha = 0.35 * composition.h2o;
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.globalAlpha = 1;
      }
      const landColor = this.#tintHex(eqT < 330 ? '#6c584c' : '#8d5524', starRGB, 0.1);
      for (let i = 0; i < 8; i++) {
        const cx = Math.random() * c.width;
        const cy = Math.random() * c.height;
        const r = 60 + Math.random() * 160;
        this.#irregularBlob(ctx, cx, cy, r, landColor);
      }
      if (eqT < 250) {
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.fillRect(0, 0, c.width, 40 + Math.random() * 40);
        ctx.fillRect(0, c.height - (40 + Math.random() * 40), c.width, 40 + Math.random() * 40);
      }
    }

    // 7) Nubes
    const thickFactor = this.#clamp01(Math.log10(1 + atmosphere.pressureBars) / 2);
    const cloudIntensity = this.#clamp01(atmosphere.cloudCover * 0.7 + thickFactor * 0.6);
    for (let i = 0; i < Math.floor(400 * cloudIntensity); i++) {
      const x = Math.random() * c.width;
      const y = Math.random() * c.height;
      const r = 4 + Math.random() * (resolvedType === 'Gaseoso' ? 18 : 10);
      ctx.beginPath();
      ctx.fillStyle = `rgba(255,255,255,${0.06 + 0.18 * Math.random() * cloudIntensity})`;
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // 8) Bruma
    if (atmosphere.haze > 0.01) {
      const hazeColor = composition.na > 0.2 ? '255,225,120' : '200,200,200';
      ctx.fillStyle = `rgba(${hazeColor},${0.08 + 0.25 * atmosphere.haze + 0.2 * thickFactor})`;
      ctx.fillRect(0, 0, c.width, c.height);
    }

    // 9) Efecto silicato en climas muy calientes
    if (composition.silicates > 0.2 && eqT > 800) {
      ctx.fillStyle = 'rgba(80,140,255,0.10)';
      ctx.fillRect(0, 0, c.width, c.height);
    }

    return c;
  }

  // Helpers privados
  #clamp01(x) { return Math.min(1, Math.max(0, x)); }
  #mix(a, b, t) { return a + (b - a) * t; }

  #hexToRgb(hex) {
    const n = hex.replace('#', '');
    const bigint = parseInt(n, 16);
    return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
    }

  #rgbToHex(r, g, b) {
    const toHex = v => ('0' + v.toString(16)).slice(-2);
    return '#' + toHex(r) + toHex(g) + toHex(b);
  }

  #tintHex(hex, rgb, amt) {
    const c = this.#hexToRgb(hex);
    const r = Math.round(this.#mix(c.r, Math.round(rgb.r * 255), amt));
    const g = Math.round(this.#mix(c.g, Math.round(rgb.g * 255), amt));
    const b = Math.round(this.#mix(c.b, Math.round(rgb.b * 255), amt));
    return this.#rgbToHex(r, g, b);
  }

  #irregularBlob(ctx, cx, cy, radius, color) {
    ctx.beginPath();
    const points = 6 + Math.floor(Math.random() * 6);
    for (let i = 0; i <= points; i++) {
      const ang = (i / points) * Math.PI * 2;
      const r = radius * (0.6 + Math.random() * 0.6);
      const x = cx + Math.cos(ang) * r;
      const y = cy + Math.sin(ang) * r * this.#mix(0.6, 1.4, Math.random());
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  #basePalette(eqT) {
    if (eqT < 250) {
      return { base1: '#88ccee', base2: '#e6f3ff' }; // helado
    } else if (eqT < 350) {
      return { base1: '#2f5d62', base2: '#a7c957' }; // templado
    }
    return { base1: '#7a2e0e', base2: '#f29f05' }; // caliente
  }

  #shiftByComposition(hex, composition, eqT) {
    const { h2o = 0, co2 = 0, ch4 = 0, na = 0, silicates = 0 } = composition || {};
    const mixRgb = {
      r: this.#clamp01(0.1 * na + (eqT > 800 ? 0.05 * silicates : 0)),
      g: this.#clamp01(0.1 * na + 0.1 * h2o + 0.2 * ch4),
      b: this.#clamp01(0.4 * h2o + 0.4 * ch4 + (eqT > 800 ? 0.4 * silicates : 0)),
    };
    let col = this.#tintHex(hex, mixRgb, 0.6);
    if (co2 > 0) {
      const c0 = this.#hexToRgb(col);
      const k = this.#clamp01(0.3 * co2);
      const r = Math.round(this.#mix(c0.r, 255, k));
      const g = Math.round(this.#mix(c0.g, 255, k));
      const b = Math.round(this.#mix(c0.b, 255, k));
      col = this.#rgbToHex(r, g, b);
    }
    return col;
  }
}
