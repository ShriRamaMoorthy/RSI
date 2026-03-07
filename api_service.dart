// lib/services/api_service.dart
// ============================================================
// Handles all communication with the FastAPI backend
// ============================================================

import 'dart:io';
import 'dart:convert';
import 'package:dio/dio.dart';

//  Config 
// Change this to your server's IP when testing on physical device
// e.g. 'http://192.168.1.100:8000'
const String kBaseUrl = 'http://10.0.2.2:8000'; // Android emulator → localhost

// ─ Models 
class Detection {
  final String className;
  final double confidence;
  final List<double> bbox;
  final bool isAlert;

  Detection({
    required this.className,
    required this.confidence,
    required this.bbox,
    required this.isAlert,
  });

  factory Detection.fromJson(Map<String, dynamic> j) => Detection(
    className:  j['class_name'],
    confidence: (j['confidence'] as num).toDouble(),
    bbox:       List<double>.from(j['bbox'].map((v) => (v as num).toDouble())),
    isAlert:    j['is_alert'] as bool,
  );
}

class ShelfAnalysis {
  final int scanId;
  final int totalSlots;
  final int emptySlots;
  final int misplacedItems;
  final int productsDetected;
  final double stockLevelPct;
  final List<String> alerts;
  final List<Detection> detections;
  final String annotatedImageB64;
  final double inferenceMs;
  final DateTime timestamp;

  ShelfAnalysis({
    required this.scanId,
    required this.totalSlots,
    required this.emptySlots,
    required this.misplacedItems,
    required this.productsDetected,
    required this.stockLevelPct,
    required this.alerts,
    required this.detections,
    required this.annotatedImageB64,
    required this.inferenceMs,
    required this.timestamp,
  });

  factory ShelfAnalysis.fromJson(Map<String, dynamic> j) => ShelfAnalysis(
    scanId:            j['scan_id'],
    totalSlots:        j['total_slots'],
    emptySlots:        j['empty_slots'],
    misplacedItems:    j['misplaced_items'],
    productsDetected:  j['products_detected'],
    stockLevelPct:     (j['stock_level_pct'] as num).toDouble(),
    alerts:            List<String>.from(j['alerts']),
    detections:        (j['detections'] as List)
                          .map((d) => Detection.fromJson(d))
                          .toList(),
    annotatedImageB64: j['annotated_image_b64'],
    inferenceMs:       (j['inference_ms'] as num).toDouble(),
    timestamp:         DateTime.parse(j['timestamp']),
  );

  bool get hasAlerts =>
    emptySlots > 0 || misplacedItems > 0 || stockLevelPct < 60;

  String get stockStatus {
    if (stockLevelPct >= 80) return 'Good';
    if (stockLevelPct >= 50) return 'Low';
    return 'Critical';
  }
}

class ScanHistoryItem {
  final int id;
  final String? storeId;
  final String? aisle;
  final double stockLevelPct;
  final int emptySlots;
  final int misplacedItems;
  final bool hasAlerts;
  final double inferenceMs;
  final DateTime createdAt;

  ScanHistoryItem({
    required this.id,
    this.storeId,
    this.aisle,
    required this.stockLevelPct,
    required this.emptySlots,
    required this.misplacedItems,
    required this.hasAlerts,
    required this.inferenceMs,
    required this.createdAt,
  });

  factory ScanHistoryItem.fromJson(Map<String, dynamic> j) => ScanHistoryItem(
    id:             j['id'],
    storeId:        j['store_id'],
    aisle:          j['aisle'],
    stockLevelPct:  (j['stock_level_pct'] as num).toDouble(),
    emptySlots:     j['empty_slots'],
    misplacedItems: j['misplaced_items'],
    hasAlerts:      j['has_alerts'],
    inferenceMs:    (j['inference_ms'] as num).toDouble(),
    createdAt:      DateTime.parse(j['created_at']),
  );
}

class AnalyticsSummary {
  final int totalScans;
  final double avgStockLevelPct;
  final double avgEmptySlots;
  final double avgMisplaced;
  final int totalAlerts;

  AnalyticsSummary({
    required this.totalScans,
    required this.avgStockLevelPct,
    required this.avgEmptySlots,
    required this.avgMisplaced,
    required this.totalAlerts,
  });

  factory AnalyticsSummary.fromJson(Map<String, dynamic> j) => AnalyticsSummary(
    totalScans:      j['total_scans'],
    avgStockLevelPct:(j['avg_stock_level_pct'] as num?)?.toDouble() ?? 0,
    avgEmptySlots:   (j['avg_empty_slots'] as num?)?.toDouble() ?? 0,
    avgMisplaced:    (j['avg_misplaced'] as num?)?.toDouble() ?? 0,
    totalAlerts:     j['total_alerts'] ?? 0,
  );
}


//  API Service 
class ApiService {
  late final Dio _dio;

  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: kBaseUrl,
      connectTimeout: const Duration(seconds: 15),
      receiveTimeout: const Duration(seconds: 60),  // inference can take time
      headers: {'Accept': 'application/json'},
    ));

    // Request/response logger (remove in production)
    _dio.interceptors.add(LogInterceptor(
      requestBody: false,
      responseBody: false,
      logPrint: (o) => debugPrint('[API] $o'),
    ));
  }

  /// Upload image for shelf analysis
  Future<ShelfAnalysis> analyzeShelf({
    required File imageFile,
    String? storeId,
    String? aisle,
    double confThreshold = 0.35,
  }) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        imageFile.path,
        filename: 'shelf_image.jpg',
      ),
      if (storeId != null) 'store_id': storeId,
      if (aisle != null) 'aisle': aisle,
      'conf_threshold': confThreshold.toString(),
    });

    try {
      final response = await _dio.post('/analyze', data: formData);
      return ShelfAnalysis.fromJson(response.data);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  /// Get scan history
  Future<List<ScanHistoryItem>> getHistory({
    int limit = 50,
    String? storeId,
  }) async {
    try {
      final response = await _dio.get('/history', queryParameters: {
        'limit': limit,
        if (storeId != null) 'store_id': storeId,
      });
      return (response.data as List)
          .map((j) => ScanHistoryItem.fromJson(j))
          .toList();
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  /// Get single scan detail
  Future<Map<String, dynamic>> getScanDetail(int scanId) async {
    try {
      final response = await _dio.get('/history/$scanId');
      return response.data;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  /// Get analytics summary
  Future<AnalyticsSummary> getAnalytics({String? storeId}) async {
    try {
      final response = await _dio.get('/analytics', queryParameters: {
        if (storeId != null) 'store_id': storeId,
      });
      return AnalyticsSummary.fromJson(response.data);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  /// Health check
  Future<bool> isHealthy() async {
    try {
      final response = await _dio.get('/health');
      return response.data['status'] == 'ok';
    } catch (_) {
      return false;
    }
  }

  String _handleError(DioException e) {
    if (e.type == DioExceptionType.connectionTimeout ||
        e.type == DioExceptionType.receiveTimeout) {
      return 'Connection timed out. Is the server running?';
    }
    if (e.type == DioExceptionType.connectionError) {
      return 'Cannot reach server. Check your network or server URL.';
    }
    final msg = e.response?.data?['detail'] ?? e.message ?? 'Unknown error';
    return 'Error: $msg';
  }
}

// Singleton for easy access
final apiService = ApiService();

// Helper
void debugPrint(String s) => print(s); // replace with logger in prod