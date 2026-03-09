// lib/screens/result_screen.dart
// ============================================================
// Result Screen — Shows annotated image + analysis breakdown
// ============================================================

import 'dart:convert';
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../theme.dart';

class ResultScreen extends StatelessWidget {
  final ShelfAnalysis analysis;

  const ResultScreen({super.key, required this.analysis});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Analysis Result'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: Text(
                '${analysis.inferenceMs.toInt()}ms',
                style: const TextStyle(
                  color: AppColors.accent,
                  fontSize: 13,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            //  Annotated image 
            _AnnotatedImageView(b64: analysis.annotatedImageB64),

            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  //  Status banner 
                  _StatusBanner(analysis: analysis),
                  const SizedBox(height: 20),

                  //  Stats grid 
                  _StatsGrid(analysis: analysis),
                  const SizedBox(height: 20),

                  //  Stock gauge 
                  _StockGauge(pct: analysis.stockLevelPct),
                  const SizedBox(height: 20),

                  //  Alerts 
                  _AlertsList(alerts: analysis.alerts),
                  const SizedBox(height: 20),

                  //  Detections list 
                  _DetectionList(detections: analysis.detections),
                  const SizedBox(height: 20),

                  //  Scan metadata 
                  _MetaCard(analysis: analysis),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Annotated Image 
class _AnnotatedImageView extends StatelessWidget {
  final String b64;
  const _AnnotatedImageView({required this.b64});

  @override
  Widget build(BuildContext context) {
    final imageBytes = base64Decode(b64);
    return Container(
      constraints: const BoxConstraints(maxHeight: 320),
      color: Colors.black,
      child: InteractiveViewer(
        child: Image.memory(imageBytes, fit: BoxFit.contain),
      ),
    );
  }
}

//  Status Banner 
class _StatusBanner extends StatelessWidget {
  final ShelfAnalysis analysis;
  const _StatusBanner({required this.analysis});

  @override
  Widget build(BuildContext context) {
    final Color color;
    final String icon;
    final String label;

    if (analysis.stockLevelPct >= 80 && !analysis.hasAlerts) {
      color = AppColors.success;
      icon = '✅';
      label = 'Shelf Compliant';
    } else if (analysis.stockLevelPct >= 50) {
      color = AppColors.warning;
      icon = '⚠️';
      label = 'Attention Required';
    } else {
      color = AppColors.danger;
      icon = '🔴';
      label = 'Critical — Restock Now';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Row(
        children: [
          Text(icon, style: const TextStyle(fontSize: 22)),
          const SizedBox(width: 12),
          Expanded(
            child: Text(label,
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.w700,
                fontSize: 16,
              ),
            ),
          ),
          Text('${analysis.stockLevelPct.toInt()}%',
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w900,
              fontSize: 22,
            ),
          ),
        ],
      ),
    );
  }
}

// Stats Grid 
class _StatsGrid extends StatelessWidget {
  final ShelfAnalysis analysis;
  const _StatsGrid({required this.analysis});

  @override
  Widget build(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.6,
      children: [
        _StatCard(
          label: 'Total Slots',
          value: '${analysis.totalSlots}',
          icon: Icons.grid_view_rounded,
          color: AppColors.info,
        ),
        _StatCard(
          label: 'Products',
          value: '${analysis.productsDetected}',
          icon: Icons.inventory_2_rounded,
          color: AppColors.success,
        ),
        _StatCard(
          label: 'Empty Slots',
          value: '${analysis.emptySlots}',
          icon: Icons.remove_shopping_cart_rounded,
          color: analysis.emptySlots > 0 ? AppColors.danger : AppColors.success,
        ),
        _StatCard(
          label: 'Misplaced',
          value: '${analysis.misplacedItems}',
          icon: Icons.swap_vert_rounded,
          color: analysis.misplacedItems > 0 ? AppColors.warning : AppColors.success,
        ),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;

  const _StatCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Icon(icon, color: color, size: 20),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(value,
                style: TextStyle(
                  color: color,
                  fontSize: 28,
                  fontWeight: FontWeight.w800,
                  height: 1,
                ),
              ),
              const SizedBox(height: 2),
              Text(label,
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// Stock Gauge
class _StockGauge extends StatelessWidget {
  final double pct;
  const _StockGauge({required this.pct});

  @override
  Widget build(BuildContext context) {
    final color = pct >= 80
        ? AppColors.success
        : pct >= 50
            ? AppColors.warning
            : AppColors.danger;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Stock Level',
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15)),
              Text('${pct.toInt()}%',
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.w800,
                  fontSize: 16,
                )),
            ],
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: pct / 100,
              backgroundColor: AppColors.surfaceLight,
              valueColor: AlwaysStoppedAnimation(color),
              minHeight: 12,
            ),
          ),
          const SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _GaugeLegend(color: AppColors.danger, label: 'Critical (<50%)'),
              _GaugeLegend(color: AppColors.warning, label: 'Low (50-80%)'),
              _GaugeLegend(color: AppColors.success, label: 'Good (>80%)'),
            ],
          ),
        ],
      ),
    );
  }
}

class _GaugeLegend extends StatelessWidget {
  final Color color;
  final String label;
  const _GaugeLegend({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8, height: 8,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text(label,
          style: const TextStyle(fontSize: 10, color: AppColors.textMuted)),
      ],
    );
  }
}

// Alerts List
class _AlertsList extends StatelessWidget {
  final List<String> alerts;
  const _AlertsList({required this.alerts});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Alerts',
          style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
        const SizedBox(height: 10),
        ...alerts.map((a) {
          final isOk = a.startsWith('✅');
          final isWarning = a.startsWith('🟡');
          final color = isOk ? AppColors.success
              : isWarning ? AppColors.warning
              : AppColors.danger;

          return Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: color.withOpacity(0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Text(a,
              style: TextStyle(color: color, fontSize: 13,
                fontWeight: FontWeight.w500)),
          );
        }),
      ],
    );
  }
}

// Detection List
class _DetectionList extends StatelessWidget {
  final List<Detection> detections;
  const _DetectionList({required this.detections});

  static const _classColors = {
    'product':    AppColors.success,
    'empty_slot': AppColors.danger,
    'misplaced':  AppColors.warning,
    'price_tag':  AppColors.info,
  };

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text('Detections',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
            Text('${detections.length} items',
              style: const TextStyle(color: AppColors.textMuted, fontSize: 13)),
          ],
        ),
        const SizedBox(height: 10),
        ...detections.take(20).map((d) {
          final color = _classColors[d.className] ?? AppColors.textSecondary;
          return Container(
            margin: const EdgeInsets.only(bottom: 6),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(
              children: [
                Container(
                  width: 8, height: 8,
                  decoration: BoxDecoration(
                    color: color, shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    d.className.replaceAll('_', ' ').toUpperCase(),
                    style: TextStyle(
                      color: color,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
                Text(
                  '${(d.confidence * 100).toInt()}%',
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 13,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
          );
        }),
        if (detections.length > 20)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              '+ ${detections.length - 20} more detections',
              style: const TextStyle(color: AppColors.textMuted, fontSize: 12),
            ),
          ),
      ],
    );
  }
}

// Meta Card
class _MetaCard extends StatelessWidget {
  final ShelfAnalysis analysis;
  const _MetaCard({required this.analysis});

  @override
  Widget build(BuildContext context) {
    final local = analysis.timestamp.toLocal();
    final timeStr =
        '${local.year}-${local.month.toString().padLeft(2, '0')}-${local.day.toString().padLeft(2, '0')} '
        '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: [
          _MetaRow(label: 'Scan ID', value: '#${analysis.scanId}'),
          _MetaRow(label: 'Timestamp', value: timeStr),
          _MetaRow(label: 'Inference Time',
            value: '${analysis.inferenceMs.toInt()}ms'),
          _MetaRow(label: 'Model Conf. Threshold', value: '35%', isLast: true),
        ],
      ),
    );
  }
}

class _MetaRow extends StatelessWidget {
  final String label;
  final String value;
  final bool isLast;

  const _MetaRow({
    required this.label,
    required this.value,
    this.isLast = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                style: const TextStyle(
                  color: AppColors.textSecondary, fontSize: 13)),
              Text(value,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                  fontFamily: 'monospace',
                )),
            ],
          ),
        ),
        if (!isLast) const Divider(height: 1),
      ],
    );
  }
}