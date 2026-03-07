// lib/screens/scan_screen.dart
// ============================================================
// Scan Screen — Upload or capture shelf image for analysis
// ============================================================

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../theme.dart';
import '../widgets/stat_chip.dart';
import 'result_screen.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with TickerProviderStateMixin {
  File? _selectedImage;
  bool _isAnalyzing = false;
  String? _error;

  final _storeController = TextEditingController();
  final _aisleController = TextEditingController();
  double _confThreshold = 0.35;

  late final AnimationController _pulseCtrl;
  late final Animation<double> _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _pulseAnim = Tween(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _storeController.dispose();
    _aisleController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final xFile = await picker.pickImage(
      source: source,
      imageQuality: 90,
      maxWidth: 1280,
    );
    if (xFile == null) return;
    setState(() {
      _selectedImage = File(xFile.path);
      _error = null;
    });
  }

  Future<void> _analyze() async {
    if (_selectedImage == null) return;

    setState(() { _isAnalyzing = true; _error = null; });

    try {
      final result = await apiService.analyzeShelf(
        imageFile: _selectedImage!,
        storeId: _storeController.text.isEmpty ? null : _storeController.text,
        aisle: _aisleController.text.isEmpty ? null : _aisleController.text,
        confThreshold: _confThreshold,
      );

      if (!mounted) return;

      await Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ResultScreen(analysis: result)),
      );

      // Reset after returning
      setState(() { _selectedImage = null; });

    } catch (e) {
      setState(() { _error = e.toString(); });
    } finally {
      setState(() { _isAnalyzing = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Shelf Scan'),
        actions: [
          IconButton(
            icon: const Icon(Icons.tune_rounded),
            tooltip: 'Settings',
            onPressed: _showSettings,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            //  Header 
            _buildHeader(),
            const SizedBox(height: 24),

            //  Image picker area 
            _buildImagePickerArea(),
            const SizedBox(height: 24),

            //  Optional metadata 
            _buildMetadataFields(),
            const SizedBox(height: 24),

            //  Error 
            if (_error != null) _buildError(),

            //  Analyze button 
            _buildAnalyzeButton(),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Analyze Shelf',
          style: Theme.of(context).textTheme.displayMedium),
        const SizedBox(height: 6),
        Text('Capture or upload a shelf photo to detect\nempty slots and compliance issues.',
          style: Theme.of(context).textTheme.bodyMedium),
      ],
    );
  }

  Widget _buildImagePickerArea() {
    return GestureDetector(
      onTap: () => _showPickerOptions(),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        height: 260,
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: _selectedImage != null
                ? AppColors.accent
                : AppColors.border,
            width: _selectedImage != null ? 2 : 1,
          ),
        ),
        child: _selectedImage != null
            ? _buildImagePreview()
            : _buildPickerPlaceholder(),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Stack(
      fit: StackFit.expand,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(19),
          child: Image.file(_selectedImage!, fit: BoxFit.cover),
        ),
        // Change image overlay
        Positioned(
          bottom: 12, right: 12,
          child: Material(
            color: AppColors.background.withOpacity(0.8),
            borderRadius: BorderRadius.circular(10),
            child: InkWell(
              onTap: _showPickerOptions,
              borderRadius: BorderRadius.circular(10),
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.swap_horiz_rounded,
                      color: AppColors.accent, size: 16),
                    SizedBox(width: 6),
                    Text('Change', style: TextStyle(
                      color: AppColors.accent,
                      fontSize: 13, fontWeight: FontWeight.w600,
                    )),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPickerPlaceholder() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        ScaleTransition(
          scale: _pulseAnim,
          child: Container(
            width: 72, height: 72,
            decoration: BoxDecoration(
              color: AppColors.accent.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: AppColors.accent.withOpacity(0.3)),
            ),
            child: const Icon(Icons.camera_alt_rounded,
              color: AppColors.accent, size: 32),
          ),
        ),
        const SizedBox(height: 16),
        Text('Tap to capture or upload',
          style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 6),
        Text('JPEG · PNG · up to 10MB',
          style: Theme.of(context).textTheme.labelSmall),
        const SizedBox(height: 16),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _SourceButton(
              icon: Icons.camera_rounded,
              label: 'Camera',
              onTap: () => _pickImage(ImageSource.camera),
            ),
            const SizedBox(width: 12),
            _SourceButton(
              icon: Icons.photo_library_rounded,
              label: 'Gallery',
              onTap: () => _pickImage(ImageSource.gallery),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildMetadataFields() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Optional Info',
          style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _storeController,
                decoration: const InputDecoration(
                  labelText: 'Store ID',
                  hintText: 'e.g. STORE-01',
                  prefixIcon: Icon(Icons.store_rounded, size: 18),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextField(
                controller: _aisleController,
                decoration: const InputDecoration(
                  labelText: 'Aisle',
                  hintText: 'e.g. A3',
                  prefixIcon: Icon(Icons.grid_view_rounded, size: 18),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildError() {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.danger.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.danger.withOpacity(0.4)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: AppColors.danger, size: 18),
          const SizedBox(width: 10),
          Expanded(child: Text(_error!,
            style: const TextStyle(color: AppColors.danger, fontSize: 13))),
        ],
      ),
    );
  }

  Widget _buildAnalyzeButton() {
    final canAnalyze = _selectedImage != null && !_isAnalyzing;
    return SizedBox(
      height: 56,
      child: ElevatedButton(
        onPressed: canAnalyze ? _analyze : null,
        style: ElevatedButton.styleFrom(
          backgroundColor: canAnalyze ? AppColors.accent : AppColors.border,
          foregroundColor: canAnalyze ? AppColors.background : AppColors.textMuted,
        ),
        child: _isAnalyzing
            ? const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                    width: 20, height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      valueColor: AlwaysStoppedAnimation(AppColors.background),
                    ),
                  ),
                  SizedBox(width: 12),
                  Text('Analyzing...', style: TextStyle(fontSize: 16)),
                ],
              )
            : const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.auto_awesome_rounded, size: 20),
                  SizedBox(width: 10),
                  Text('Analyze Shelf', style: TextStyle(fontSize: 16)),
                ],
              ),
      ),
    );
  }

  void _showPickerOptions() {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 36, height: 4,
              decoration: BoxDecoration(
                color: AppColors.border,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 20),
            Text('Select Image Source',
              style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 20),
            ListTile(
              leading: const Icon(Icons.camera_rounded, color: AppColors.accent),
              title: const Text('Camera'),
              subtitle: const Text('Take a new photo'),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo_library_rounded,
                color: AppColors.accent),
              title: const Text('Photo Library'),
              subtitle: const Text('Choose existing image'),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showSettings() {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => StatefulBuilder(
        builder: (ctx, setS) => Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Detection Settings',
                style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Confidence Threshold'),
                  Text('${(_confThreshold * 100).toInt()}%',
                    style: const TextStyle(color: AppColors.accent,
                      fontWeight: FontWeight.w700)),
                ],
              ),
              Slider(
                value: _confThreshold,
                min: 0.1, max: 0.9, divisions: 16,
                activeColor: AppColors.accent,
                onChanged: (v) {
                  setS(() => _confThreshold = v);
                  setState(() => _confThreshold = v);
                },
              ),
              const Text(
                'Lower = more detections (may include false positives)\n'
                'Higher = fewer, more confident detections',
                style: TextStyle(color: AppColors.textMuted, fontSize: 12),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}

//  Small source button widget 
class _SourceButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _SourceButton({
    required this.icon, required this.label, required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: AppColors.surfaceLight,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          children: [
            Icon(icon, size: 16, color: AppColors.accent),
            const SizedBox(width: 6),
            Text(label, style: const TextStyle(
              fontSize: 13, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}