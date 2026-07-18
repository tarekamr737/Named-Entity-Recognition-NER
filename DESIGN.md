---
name: Obsidian Precision
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1b1b1b'
  surface-container: '#1f1f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353535'
  on-surface: '#e2e2e2'
  on-surface-variant: '#b9cacb'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#303030'
  outline: '#849495'
  outline-variant: '#3b494b'
  surface-tint: '#00dbe9'
  primary: '#dbfcff'
  on-primary: '#00363a'
  primary-container: '#00f0ff'
  on-primary-container: '#006970'
  inverse-primary: '#006970'
  secondary: '#d1bcff'
  on-secondary: '#3c0090'
  secondary-container: '#7000ff'
  on-secondary-container: '#ddcdff'
  tertiary: '#fff3f4'
  on-tertiary: '#66002c'
  tertiary-container: '#ffccd6'
  on-tertiary-container: '#bb0058'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#7df4ff'
  primary-fixed-dim: '#00dbe9'
  on-primary-fixed: '#002022'
  on-primary-fixed-variant: '#004f54'
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#d1bcff'
  on-secondary-fixed: '#23005b'
  on-secondary-fixed-variant: '#5700c9'
  tertiary-fixed: '#ffd9e0'
  tertiary-fixed-dim: '#ffb1c3'
  on-tertiary-fixed: '#3f0019'
  on-tertiary-fixed-variant: '#8f0041'
  background: '#131313'
  on-background: '#e2e2e2'
  surface-variant: '#353535'
typography:
  headline-lg:
    fontFamily: Geist
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 30px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-desktop: 40px
  margin-mobile: 16px
  container-max: 1440px
---

## Brand & Style
The brand personality is high-performance, technical, and immersive. It is designed for elite developer tools and data-intensive environments where focus is paramount. 

The design system adopts a **Glassmorphic-Cyber** aesthetic. It leverages an absolute black foundation to create infinite depth, allowing semi-transparent, frosted glass interfaces to float within the workspace. The emotional response is one of surgical precision and sophisticated control. The UI feels like a high-end command center—dark, quiet, and illuminated only by vital data points and translucent crystalline panels.

## Colors
The palette is rooted in a "True Black" (#000000) base to maximize the contrast of glassy elements. 

- **Primary (Electric Cyan):** Used for active states, primary actions, and critical data paths. It pierces through the dark background.
- **Secondary (Neon Violet):** Used for secondary features, code syntax, and decorative accents.
- **Tertiary (Plasma Pink):** Used for destructive actions or high-priority alerts.
- **Surface Tones:** Surfaces are not solid; they use a semi-transparent charcoal with a slight blue tint to simulate tinted glass.
- **Accents:** Neon colors are refined with a 1px inner glow or "edge lighting" effect to harmonize with the glass textures.

## Typography
Typography in this design system is technical and ultra-modern. **Geist** provides a clean, sans-serif foundation for high readability in dense layouts. **JetBrains Mono** is utilized for labels, metadata, and code snippets to reinforce the technical nature of the system.

- **Headlines:** Set with tight tracking and bold weights to ground the airy glass panels.
- **Labels:** Always uppercase or monospaced to denote system-level information.
- **Hierarchy:** Use color (Primary Cyan vs. Dimmed White) rather than just size to establish importance within the translucent containers.

## Layout & Spacing
The layout follows a strict 4px grid system to ensure technical alignment. 

- **Grid:** A 12-column fluid grid for desktop with wide 24px gutters to allow the "glow" of glassy elements to breathe.
- **Glass Panels:** Layout panels should never touch the edge of the screen; they should float with a minimum 16px margin on mobile and 40px on desktop, creating a "HUD" (Heads-Up Display) effect.
- **Density:** High density is encouraged, but visual separation must be maintained through backdrop blurs rather than heavy margins.

## Elevation & Depth
Depth is created through "Optic Layers" rather than traditional shadows.

1.  **Level 0 (Base):** True Black (#000000).
2.  **Level 1 (Sub-surface):** 12px Backdrop Blur, 40% opacity surface, 1px subtle border (White at 10% opacity).
3.  **Level 2 (Active/Floating):** 24px Backdrop Blur, 60% opacity surface, 1px border (Primary Cyan at 30% opacity).
4.  **Lighting:** A faint radial gradient (Primary or Secondary color at 5% opacity) can be placed behind panels to simulate a "glow" reflecting off the back of the glass. 
5.  **Shadows:** Shadows are omitted entirely or replaced with a soft 20px blur of the primary color with 0 offset.

## Shapes
Shapes are precise and architectural. A "Soft" (0.25rem - 0.75rem) roundedness is used to prevent the UI from feeling too aggressive, while maintaining a sharp, professional edge.

- **Glass Sheets:** Use `rounded-lg` (0.5rem) to soften the large panels.
- **Controls:** Use `rounded-sm` (0.25rem) for buttons and inputs to maintain a technical, "machined" look.
- **Interaction:** On hover, shapes may transition from 1px subtle borders to a slightly thicker neon-lit edge.

## Components
- **Glass Cards:** The core container. Must have `backdrop-filter: blur(12px)`, a background of `rgba(20, 20, 20, 0.6)`, and a top-down linear gradient border (White 15% to White 5%).
- **Buttons:** Primary buttons use a solid-to-transparent gradient of the Primary color with high-contrast black text. Secondary buttons are ghost-style with a glassy background and a neon border.
- **Inputs:** Darker than the cards (`rgba(0,0,0,0.4)`) with a 1px bottom border that glows Primary Cyan when focused.
- **Chips/Tags:** Monospaced text inside a pill shape with a 10% opacity fill of the respective entity color (Cyan, Violet, or Pink).
- **Lists:** Separated by 1px glass-etched lines. Hover states should trigger a subtle increase in the backdrop blur and a faint lateral glow.
- **Data Visualizations:** Lines and points should use the neon palette with "bloom" effects (soft outer glows) to simulate light emitting from within the glass.