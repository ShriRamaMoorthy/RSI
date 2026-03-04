// lib/theme.dart
// ============================================================
// App Theme — Industrial / Retail dark-accented design
// ============================================================

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  // Primary palette
  static const background   = Color(0xFF0A0E1A);   // deep navy
  static const surface      = Color(0xFF111827);   // card bg
  static const surfaceLight = Color(0xFF1F2937);   // elevated card
  static const border       = Color(0xFF374151);

  // Accent
  static const accent       = Color(0xFF00D4FF);   // electric cyan
  static const accentDim    = Color(0xFF0891B2);

  // Status
  static const success      = Color(0xFF10B981);   // green
  static const warning      = Color(0xFFF59E0B);   // amber
  static const danger       = Color(0xFFEF4444);   // red
  static const info         = Color(0xFF6366F1);   // indigo

  // Text
  static const textPrimary  = Color(0xFFF9FAFB);
  static const textSecondary= Color(0xFF9CA3AF);
  static const textMuted    = Color(0xFF4B5563);
}

class AppTheme {
  static ThemeData get dark => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.background,
    colorScheme: const ColorScheme.dark(
      primary:   AppColors.accent,
      secondary: AppColors.accentDim,
      surface:   AppColors.surface,
      error:     AppColors.danger,
    ),
    textTheme: GoogleFonts.dmSansTextTheme(ThemeData.dark().textTheme).copyWith(
      displayLarge: GoogleFonts.dmSans(
        color: AppColors.textPrimary,
        fontSize: 32, fontWeight: FontWeight.w700,
      ),
      displayMedium: GoogleFonts.dmSans(
        color: AppColors.textPrimary,
        fontSize: 24, fontWeight: FontWeight.w700,
      ),
      titleLarge: GoogleFonts.dmSans(
        color: AppColors.textPrimary,
        fontSize: 20, fontWeight: FontWeight.w600,
      ),
      titleMedium: GoogleFonts.dmSans(
        color: AppColors.textPrimary,
        fontSize: 16, fontWeight: FontWeight.w500,
      ),
      bodyLarge: GoogleFonts.dmSans(
        color: AppColors.textPrimary, fontSize: 16,
      ),
      bodyMedium: GoogleFonts.dmSans(
        color: AppColors.textSecondary, fontSize: 14,
      ),
      labelSmall: GoogleFonts.dmMono(
        color: AppColors.textMuted, fontSize: 11,
        letterSpacing: 0.5,
      ),
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.background,
      elevation: 0,
      centerTitle: false,
      titleTextStyle: TextStyle(
        color: AppColors.textPrimary,
        fontSize: 20, fontWeight: FontWeight.w700,
      ),
      iconTheme: IconThemeData(color: AppColors.textPrimary),
    ),
    cardTheme: const CardThemeData(
      color: AppColors.surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(16)),
        side: BorderSide(color: AppColors.border, width: 1),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.accent,
        foregroundColor: AppColors.background,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        textStyle: GoogleFonts.dmSans(fontWeight: FontWeight.w700, fontSize: 15),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.surfaceLight,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppColors.accent, width: 2),
      ),
      labelStyle: const TextStyle(color: AppColors.textSecondary),
      hintStyle: const TextStyle(color: AppColors.textMuted),
    ),
    dividerTheme: const DividerThemeData(color: AppColors.border),
    bottomNavigationBarTheme: const BottomNavigationBarThemeData(
      backgroundColor: AppColors.surface,
      selectedItemColor: AppColors.accent,
      unselectedItemColor: AppColors.textMuted,
      type: BottomNavigationBarType.fixed,
      elevation: 0,
    ),
  );
}