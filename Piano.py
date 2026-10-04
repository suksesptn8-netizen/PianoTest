import time
import sys
import numpy as np
import sounddevice as sd

# 1. Konfigurasi frekuensi dasar piano (Hz)
F = {
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77
}

# 2. Daftar Rumus Kombinasi Akord Lengkap (Studio Bank)
CHORDS = {
    # === AKORD MAYOR ===
    'C':   [F['C4'], F['E4'], F['G4']], 'D':   [F['D4'], F['F#4'], F['A4']], 'E':   [F['E4'], F['G#4'], F['B4']],
    'F':   [F['F4'], F['A4'], F['C5']], 'G':   [F['G4'], F['B4'], F['D5']], 'A':   [F['A4'], F['C#5'], F['E5']], 'B':   [F['B4'], F['D#5'], F['F#5']],

    # === AKORD MINOR ===
    'Cm':  [F['C4'], F['D#4'], F['G4']], 'Dm':  [F['D4'], F['F4'], F['A4']], 'Em':  [F['E4'], F['G4'], F['B4']],
    'Fm':  [F['F4'], F['G#4'], F['C5']], 'Gm':  [F['G4'], F['A#4'], F['D5']], 'Am':  [F['A4'], F['C5'], F['E5']], 'Bm':  [F['B4'], F['D5'], F['F#5']],

    # === AKORD SHARP MAYOR (#) ===
    'C#':  [F['C#4'], F['F4'], F['G#4']], 'D#':  [F['D#4'], F['G4'], F['A#4']], 'F#':  [F['F#4'], F['A#4'], F['C#5']],
    'G#':  [F['G#4'], F['C5'], F['D#5']], 'A#':  [F['A#4'], F['D5'], F['F5']],

    # === AKORD SHARP MINOR (#m) ===
    'C#m': [F['C#4'], F['E4'], F['G#4']], 'D#m': [F['D#4'], F['F#4'], F['A#4']], 'F#m': [F['F#4'], F['A4'], F['C#5']],
    'G#m': [F['G#4'], F['B4'], F['D#5']], 'A#m': [F['A#4'], F['C#5'], F['F5']],

    # === AKORD DOMINANT 7th ===
    'C7':  [F['C4'], F['E4'], F['G4'], F['A#4']], 'D7':  [F['D4'], F['F#4'], F['A4'], F['C5']], 'E7':  [F['E4'], F['G#4'], F['B4'], F['D5']],
    'F7':  [F['F4'], F['A4'], F['C5'], F['D#5']], 'G7':  [F['G4'], F['B4'], F['D5'], F['F5']], 'A7':  [F['A4'], F['C#5'], F['E5'], F['G5']],
    'B7':  [F['B4'], F['D#5'], F['F#5'], F['A5']],

    # === AKORD SHARP DOMINANT 7th (#7) ===
    'C#7': [F['C#4'], F['F4'], F['G#4'], F['B4']], 'D#7': [F['D#4'], F['G4'], F['A#4'], F['C#5']], 'F#7': [F['F#4'], F['A#4'], F['C#5'], F['E5']],
    'G#7': [F['G#4'], F['C5'], F['D#5'], F['F#5']], 'A#7': [F['A#4'], F['D5'], F['F5'], F['G#5']],

    # === AKORD SHARP MINOR 7th (#m7) ===
    'C#m7': [F['C#4'], F['E4'], F['G#4'], F['B4']], 'D#m7': [F['D#4'], F['F#4'], F['A#4'], F['C#5']], 'F#m7': [F['F#4'], F['A4'], F['C#4'], F['E4']],
    'G#m7': [F['G#4'], F['B4'], F['D#5'], F['F#5']], 'A#m7': [F['A#4'], F['C#5'], F['F5'], F['G#5']],
}

# 3. Pemetaan Tombol Keyboard Lengkap
KEY_MAP = {
    'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'a': 'A', 'b': 'B',
    'z': 'Cm', 'x': 'Dm', 'v': 'Em', 'n': 'Fm', 'm': 'Gm', 'k': 'Am', 'l': 'Bm',
    'C': 'C#', 'D': 'D#', 'F': 'F#', 'G': 'G#', 'A': 'A#',
    'Z': 'C#m', 'X': 'D#m', 'V': 'F#m', 'N': 'G#m', 'M': 'A#m',
    '1': 'C7', '2': 'D7', '3': 'E7', '4': 'F7', '5': 'G7', '6': 'A7', '7': 'B7',
    'q': 'C#7', 'w': 'D#7', 'r': 'F#7', 't': 'G#7', 'y': 'A#7',
    'Q': 'C#m7', 'W': 'D#m7', 'R': 'F#m7', 'T': 'G#m7', 'Y': 'A#m7'
}

sustain_active = False

def render_chord_with_env(frequencies, is_sustain=False, duration=3.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    combined_wave = np.zeros_like(t)
    harmonic_weights = [1.0, 0.4, 0.15, 0.05]  # Harmonis atas diperhalus agar suara bulat
    
    for freq in frequencies:
        chord_wave = np.zeros_like(t)
        for i, weight in enumerate(harmonic_weights):
            harmonic_freq = freq * (i + 1)
            if harmonic_freq < sample_rate / 2:
                chord_wave += weight * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Ketukan kayu palu piano (hammer click) diperhalus
        hammer_click = np.random.normal(0, 1, len(t)) * np.exp(-200 * t) * 0.05
        chord_wave += hammer_click
        combined_wave += chord_wave
        
    # --- PROSES ANTI-PECAH (NORMALIZATION & LIMITER) ---
    # Membagi dengan nilai absolut tertinggi agar gelombang tidak pernah menembus batas distorsi (1.0)
    max_val = np.max(np.abs(combined_wave))
    if max_val > 0:
        combined_wave = combined_wave / max_val
        
    # Skala master volume akhir diturunkan ke tingkat aman (0.4) agar saat ditumpuk suaranya tidak pecah
    combined_wave = combined_wave * 0.4
    
    decay_rate = 0.5 if is_sustain else 2.5
    envelope = np.exp(-decay_rate * t)
    combined_wave = combined_wave * envelope
    
    return combined_wave.astype(np.float32)

# --- PROSES PRE-RENDERING ---
print("Sedang memproses & memuat audio piano ke RAM... Harap tunggu sebentar.")
AUDIO_NORMAL = {}
AUDIO_SUSTAIN = {}

for chord_name, freqs in CHORDS.items():
    AUDIO_NORMAL[chord_name] = render_chord_with_env(freqs, is_sustain=False)
    AUDIO_SUSTAIN[chord_name] = render_chord_with_env(freqs, is_sustain=True)

print("-> Audio berhasil dimuat! Piano siap dimainkan.")

def print_status_bar():
    status = "AKTIF (Gema Panjang)" if sustain_active else "MATI (Normal)"
    print(f"[ STATUS SUSTAIN PEDAL: {status} ]   ", end="\r", flush=True)

print("\n=== APLIKASI PIANO AKORD PYTHON (ANTI-DISTORSI) ===")
print("Kontrol Khusus:")
print("  [ Spacebar ] -> Aktifkan / Matikan Pedal Sustain")
print("  [ Esc ]      -> Keluar dari program piano\n")
print_status_bar()

if sys.platform == "win32":
    import msvcrt
    try:
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8')
                if key == '\x1b': # Tombol Esc
                    break
                elif key == ' ': # Tombol Spacebar
                    sustain_active = not sustain_active
                    print_status_bar()
                    continue
                    
                if key in KEY_MAP:
                    target_chord = KEY_MAP[key]
                    audio_source = AUDIO_SUSTAIN if sustain_active else AUDIO_NORMAL
                    
                    # Tembakkan suara instan dari RAM
                    sd.play(audio_source[target_chord], 44100)
                    print(f"-> Memainkan Akord: {target_chord}                   ")
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass

print("\n\nAplikasi Piano ditutup. Selamat bermusik!")
