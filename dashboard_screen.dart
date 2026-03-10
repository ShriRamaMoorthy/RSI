// lib/screens/dashboard_screen.dart
// ============================================================
// Dashboard Screen — Analytics, KPIs, charts
// ============================================================

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../services/api_service.dart';
import '../theme.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  AnalyticsSummary? _summary;
  List<ScanHistoryItem> _history = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final results = await Future.wait([
        apiService.getAnalytics(),
        apiService.getHistory(limit: 30),
      ]);
      setState(() {
        _summary = results[0] as AnalyticsSummary;
        _history = results[1] as List<ScanHistoryItem>;
      });
    } catch (e) {
      setState(() { _error = e.toString(); });
    } finally {
      setState(() { _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dashboard'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            onPressed: _load,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: AppColors.accent))
          : _error != null
              ? _buildError()
              : _buildDashboard(),
    );
  }

  Widget _buildDashboard() {
    if (_summary == null) return const SizedBox.shrink();

    return RefreshIndicator(
      onRefresh: _load,
      color: AppColors.accent,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // KPI Cards 
            Text('Overview',
              style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 14),
            _KpiGrid(summary: _summary!),
            const SizedBox(height: 28),

            // Stock Level Chart 
            Text('Stock Level Trend (Last 30 Scans)',
              style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 14),
            _StockChart(history: _history),
            const SizedBox(height: 28),

            // Detection Breakdown Donut 
            Text('Detection Breakdown',
              style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 14),
            _DonutChart(summary: _summary!),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _buildError() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.cloud_off_rounded,
            size: 48, color: AppColors.textMuted),
          const SizedBox(height: 12),
          Text(_error!,
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppColors.textSecondary)),
          const SizedBox(height: 16),
          ElevatedButton(onPressed: _load, child: const Text('Retry')),
        ],
      ),
    );
  }
}

// KPI Grid
class _KpiGrid extends StatelessWidget {
  final AnalyticsSummary summary;
  const _KpiGrid({required this.summary});

  @override
  Widget build(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.45,
      children: [
        _KpiCard(
          label: 'Total Scans',
          value: '${summary.totalScans}',
          icon: Icons.qr_code_scanner_rounded,
          color: AppColors.accent,
        ),
        _KpiCard(
          label: 'Avg Stock Level',
          value: '${summary.avgStockLevelPct.toInt()}%',
          icon: Icons.inventory_rounded,
          color: summary.avgStockLevelPct >= 80
              ? AppColors.success
              : summary.avgStockLevelPct >= 50
                  ? AppColors.warning
                  : AppColors.danger,
        ),
        _KpiCard(
          label: 'Avg Empty Slots',
          value: summary.avgEmptySlots.toStringAsFixed(1),
          icon: Icons.remove_shopping_cart_rounded,
          color: AppColors.danger,
        ),
        _KpiCard(
          label: 'Total Alerts',
          value: '${summary.totalAlerts}',
          icon: Icons.notifications_rounded,
          color: AppColors.warning,
        ),
      ],
    );
  }
}

class _KpiCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;

  const _KpiCard({
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
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(value,
                style: TextStyle(
                  color: color,
                  fontSize: 26,
                  fontWeight: FontWeight.w800,
                  height: 1,
                )),
              const SizedBox(height: 3),
              Text(label,
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 12,
                )),
            ],
          ),
        ],
      ),
    );
  }
}

// Stock Level Line Chart
class _StockChart extends StatelessWidget {
  final List<ScanHistoryItem> history;
  const _StockChart({required this.history});

  @override
  Widget build(BuildContext context) {
    if (history.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(40),
          child: Text('No data yet', style: TextStyle(color: AppColors.textMuted)),
        ),
      );
    }

    final spots = history.reversed.toList().asMap().entries.map((e) {
      return FlSpot(e.key.toDouble(), e.value.stockLevelPct);
    }).toList();

    return Container(
      height: 200,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: LineChart(
        LineChartData(
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            horizontalInterval: 25,
            getDrawingHorizontalLine: (v) => FlLine(
              color: AppColors.border, strokeWidth: 0.8,
            ),
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: 25,
                reservedSize: 32,
                getTitlesWidget: (v, _) => Text(
                  '${v.toInt()}%',
                  style: const TextStyle(
                    color: AppColors.textMuted, fontSize: 10),
                ),
              ),
            ),
            bottomTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false)),
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false)),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false)),
          ),
          borderData: FlBorderData(show: false),
          minY: 0, maxY: 100,
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              curveSmoothness: 0.3,
              color: AppColors.accent,
              barWidth: 2.5,
              dotData: FlDotData(
                show: true,
                getDotPainter: (s, _, __, ___) => FlDotCirclePainter(
                  radius: 3,
                  color: AppColors.accent,
                  strokeWidth: 1.5,
                  strokeColor: AppColors.background,
                ),
              ),
              belowBarData: BarAreaData(
                show: true,
                gradient: LinearGradient(
                  colors: [
                    AppColors.accent.withOpacity(0.25),
                    AppColors.accent.withOpacity(0.0),
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Donut Chart
class _DonutChart extends StatelessWidget {
  final AnalyticsSummary summary;
  const _DonutChart({required this.summary});

  @override
  Widget build(BuildContext context) {
    final total = summary.avgStockLevelPct +
        summary.avgEmptySlots +
        summary.avgMisplaced;

    if (total == 0) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(40),
          child: Text('No data yet', style: TextStyle(color: AppColors.textMuted)),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          SizedBox(
            height: 160,
            width: 160,
            child: PieChart(PieChartData(
              sectionsSpace: 3,
              centerSpaceRadius: 50,
              sections: [
                PieChartSectionData(
                  value: summary.avgStockLevelPct,
                  color: AppColors.success,
                  radius: 30,
                  title: '',
                ),
                PieChartSectionData(
                  value: summary.avgEmptySlots * 10, // scale for visibility
                  color: AppColors.danger,
                  radius: 30,
                  title: '',
                ),
                PieChartSectionData(
                  value: summary.avgMisplaced * 10,
                  color: AppColors.warning,
                  radius: 30,
                  title: '',
                ),
              ],
            )),
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _Legend(color: AppColors.success,
                  label: 'Stocked',
                  value: '${summary.avgStockLevelPct.toInt()}%'),
                const SizedBox(height: 12),
                _Legend(color: AppColors.danger,
                  label: 'Avg Empty',
                  value: summary.avgEmptySlots.toStringAsFixed(1)),
                const SizedBox(height: 12),
                _Legend(color: AppColors.warning,
                  label: 'Avg Misplaced',
                  value: summary.avgMisplaced.toStringAsFixed(1)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  final Color color;
  final String label;
  final String value;

  const _Legend({
    required this.color,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 12, height: 12,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(label,
            style: const TextStyle(
              color: AppColors.textSecondary, fontSize: 13)),
        ),
        Text(value,
          style: const TextStyle(
            color: AppColors.textPrimary,
            fontWeight: FontWeight.w700,
            fontSize: 13,
          )),
      ],
    );
  }
}