// lib/screens/history_screen.dart
// ============================================================
// History Screen — Past scan records
// ============================================================

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../theme.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<ScanHistoryItem> _items = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() { _loading = true; _error = null; });
    try {
      final items = await apiService.getHistory(limit: 100);
      setState(() { _items = items; });
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
        title: const Text('Scan History'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            onPressed: _loadHistory,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: AppColors.accent))
          : _error != null
              ? _buildError()
              : _items.isEmpty
                  ? _buildEmpty()
                  : _buildList(),
    );
  }

  Widget _buildList() {
    return RefreshIndicator(
      onRefresh: _loadHistory,
      color: AppColors.accent,
      child: ListView.separated(
        padding: const EdgeInsets.all(16),
        itemCount: _items.length,
        separatorBuilder: (_, __) => const SizedBox(height: 10),
        itemBuilder: (_, i) => _HistoryCard(item: _items[i]),
      ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.history_rounded,
            size: 64, color: AppColors.textMuted),
          const SizedBox(height: 16),
          Text('No scans yet',
            style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 8),
          const Text('Scan a shelf to see history here.',
            style: TextStyle(color: AppColors.textSecondary)),
        ],
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
          ElevatedButton(
            onPressed: _loadHistory,
            child: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}

class _HistoryCard extends StatelessWidget {
  final ScanHistoryItem item;
  const _HistoryCard({required this.item});

  @override
  Widget build(BuildContext context) {
    final stockColor = item.stockLevelPct >= 80
        ? AppColors.success
        : item.stockLevelPct >= 50
            ? AppColors.warning
            : AppColors.danger;

    final dateStr = DateFormat('MMM d, HH:mm').format(item.createdAt.toLocal());

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Top row
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: stockColor.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '${item.stockLevelPct.toInt()}%',
                    style: TextStyle(
                      color: stockColor,
                      fontWeight: FontWeight.w800,
                      fontSize: 14,
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                if (item.hasAlerts)
                  const Icon(Icons.warning_amber_rounded,
                    color: AppColors.warning, size: 18),
                const Spacer(),
                Text('#${item.id}',
                  style: const TextStyle(
                    color: AppColors.textMuted,
                    fontSize: 12,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Stats row
            Row(
              children: [
                _MiniStat(
                  icon: Icons.remove_shopping_cart_rounded,
                  value: '${item.emptySlots}',
                  label: 'empty',
                  color: item.emptySlots > 0
                      ? AppColors.danger : AppColors.textMuted,
                ),
                const SizedBox(width: 16),
                _MiniStat(
                  icon: Icons.swap_vert_rounded,
                  value: '${item.misplacedItems}',
                  label: 'misplaced',
                  color: item.misplacedItems > 0
                      ? AppColors.warning : AppColors.textMuted,
                ),
                const Spacer(),
                // Time + location
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(dateStr,
                      style: const TextStyle(
                        color: AppColors.textSecondary, fontSize: 12)),
                    if (item.aisle != null)
                      Text('Aisle ${item.aisle}',
                        style: const TextStyle(
                          color: AppColors.textMuted, fontSize: 11)),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MiniStat extends StatelessWidget {
  final IconData icon;
  final String value;
  final String label;
  final Color color;

  const _MiniStat({
    required this.icon,
    required this.value,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 4),
        Text(value,
          style: TextStyle(
            color: color, fontSize: 13, fontWeight: FontWeight.w700)),
        const SizedBox(width: 3),
        Text(label,
          style: const TextStyle(
            color: AppColors.textMuted, fontSize: 12)),
      ],
    );
  }
}