/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'baymax-bg': '#0f1419',
        'baymax-card': '#1a1f25',
        'baymax-border': '#2d3741',
        'baymax-primary': '#3b82f6',
        'baymax-success': '#22c55e',
        'baymax-warning': '#eab308',
        'baymax-danger': '#ef4444',
        'baymax-info': '#06b6d4',
      },
      animation: {
        'blink': 'blink 4s ease-in-out infinite',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      },
      keyframes: {
        blink: {
          '0%, 90%, 100%': { transform: 'scaleY(1)' },
          '95%': { transform: 'scaleY(0.1)' },
        },
      },
    },
  },
  plugins: [],
}
