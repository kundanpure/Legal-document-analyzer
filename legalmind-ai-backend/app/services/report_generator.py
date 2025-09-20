"""
PDF report generation service using ReportLab with comprehensive fallbacks
"""

import asyncio
import io
import base64
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from io import BytesIO
from dotenv import load_dotenv
load_dotenv() 
from config.logging import get_logger

logger = get_logger(__name__)

# Try to import dependencies with fallbacks
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white, red, green, orange
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus import Image as ReportLabImage
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
    logger.info("✅ ReportLab available - full PDF generation enabled")
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("⚠️ ReportLab not available - using fallback report generation")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
    logger.info("✅ Matplotlib available - chart generation enabled")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("⚠️ Matplotlib not available - charts will be text-based")

# Try to import settings and storage with fallbacks
try:
    from config.settings import get_settings
    settings = get_settings()
except ImportError:
    logger.warning("⚠️ Settings not available - using defaults")
    settings = type('MockSettings', (), {})()

try:
    from app.services.storage_manager import StorageManager
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.warning("⚠️ StorageManager not available - using local file operations")

# Custom exceptions
class ReportGenerationError(Exception):
    """Exception raised for report generation errors"""
    pass

# Mock storage manager
class MockStorageManager:
    async def save_report_pdf(self, pdf_data: bytes, report_id: str, filename: str) -> str:
        """Mock save operation - returns local file path"""
        try:
            filepath = f"reports/{filename}"
            # In production, save to local filesystem or return a mock URL
            return f"/downloads/{filename}"
        except Exception as e:
            logger.error(f"Mock storage save failed: {e}")
            return f"/mock-reports/{filename}"

class ReportGenerator:
    """Professional PDF report generation service with comprehensive fallbacks"""
    
    def __init__(self):
        self.logger = logger
        self.storage_manager = StorageManager() if STORAGE_AVAILABLE else MockStorageManager()
        
        # Define color scheme
        if REPORTLAB_AVAILABLE:
            self.colors = {
                'primary': HexColor('#1a73e8'),
                'secondary': HexColor('#34a853'),
                'warning': HexColor('#fbbc04'),
                'danger': HexColor('#ea4335'),
                'dark': HexColor('#202124'),
                'light': HexColor('#f8f9fa'),
                'border': HexColor('#e8eaed')
            }
        else:
            self.colors = {
                'primary': '#1a73e8',
                'secondary': '#34a853',
                'warning': '#fbbc04',
                'danger': '#ea4335',
                'dark': '#202124',
                'light': '#f8f9fa',
                'border': '#e8eaed'
            }
        
        # Initialize styles
        if REPORTLAB_AVAILABLE:
            self.styles = self._create_custom_styles()
        else:
            self.styles = {}
        
        self.logger.info(f"ReportGenerator initialized - ReportLab: {'✓' if REPORTLAB_AVAILABLE else '✗'}, Charts: {'✓' if MATPLOTLIB_AVAILABLE else '✗'}")

    async def generate_comprehensive_report(
        self,
        document_data: Dict[str, Any],
        report_type: str = "comprehensive",
        language: str = "en",
        include_charts: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive legal document analysis report
        
        Args:
            document_data: Document analysis data
            report_type: Type of report (brief, comprehensive, risk_focused)
            language: Report language
            include_charts: Whether to include charts
            
        Returns:
            Report generation result
        """
        
        try:
            self.logger.info(f"Generating {report_type} report in {language}")
            
            if REPORTLAB_AVAILABLE:
                return await self._generate_pdf_report(document_data, report_type, language, include_charts)
            else:
                return await self._generate_text_report(document_data, report_type, language, include_charts)
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise ReportGenerationError(f"Failed to generate report: {str(e)}")

    async def _generate_pdf_report(
        self,
        document_data: Dict[str, Any],
        report_type: str,
        language: str,
        include_charts: bool
    ) -> Dict[str, Any]:
        """Generate PDF report using ReportLab"""
        
        # Create PDF buffer
        buffer = BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        
        # Build report content
        story = []
        
        # Add title page
        story.extend(self._create_title_page(document_data, report_type))
        story.append(PageBreak())
        
        # Add executive summary
        story.extend(self._create_executive_summary(document_data))
        story.append(Spacer(1, 20))
        
        # Add risk analysis section
        story.extend(self._create_risk_analysis_section(document_data, include_charts))
        story.append(Spacer(1, 20))
        
        # Add key findings section
        story.extend(self._create_key_findings_section(document_data))
        story.append(Spacer(1, 20))
        
        # Add recommendations section
        story.extend(self._create_recommendations_section(document_data))
        story.append(Spacer(1, 20))
        
        # Add detailed analysis (if comprehensive)
        if report_type == "comprehensive":
            story.extend(self._create_detailed_analysis_section(document_data))
            story.append(Spacer(1, 20))
        
        # Add appendix
        story.extend(self._create_appendix_section(document_data))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Save to storage
        report_id = document_data.get('document_id', f'report_{uuid.uuid4().hex[:8]}')
        download_url = await self.storage_manager.save_report_pdf(
            pdf_data, report_id, f"legal_analysis_{report_id}.pdf"
        )
        
        # Create report metadata
        report_data = {
            'report_id': report_id,
            'title': f"Legal Document Analysis - {document_data.get('title', 'Document')}",
            'document_type': document_data.get('document_type', 'Unknown'),
            'risk_score': document_data.get('overall_risk_score', 0),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'language': language,
            'report_type': report_type,
            'page_count': len(story) // 10,  # Rough estimate
            'file_size': len(pdf_data),
            'sections': [
                'Executive Summary',
                'Risk Analysis',
                'Key Findings',
                'Recommendations',
                'Detailed Analysis' if report_type == "comprehensive" else None,
                'Appendix'
            ],
            'charts_included': include_charts,
            'format': 'PDF'
        }
        
        self.logger.info(f"PDF Report generated successfully: {report_id}")
        
        return {
            'report_data': report_data,
            'download_url': download_url,
            'pdf_size': len(pdf_data)
        }

    async def _generate_text_report(
        self,
        document_data: Dict[str, Any],
        report_type: str,
        language: str,
        include_charts: bool
    ) -> Dict[str, Any]:
        """Generate text-based report when ReportLab is not available"""
        
        report_sections = []
        
        # Title section
        title = f"LEGAL DOCUMENT ANALYSIS REPORT\n{'=' * 50}\n"
        title += f"Document: {document_data.get('title', 'Legal Document')}\n"
        title += f"Type: {document_data.get('document_type', 'Unknown').replace('_', ' ').title()}\n"
        title += f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}\n"
        title += f"Report Type: {report_type.title()}\n"
        title += f"Risk Score: {document_data.get('overall_risk_score', 0):.1f}/10.0\n\n"
        
        report_sections.append(title)
        
        # Executive Summary
        summary_section = "EXECUTIVE SUMMARY\n" + "-" * 20 + "\n"
        summary = document_data.get('summary', 'No summary available.')
        summary_section += f"{summary}\n\n"
        
        # Key metrics
        risk_score = document_data.get('overall_risk_score', 0)
        risk_level = 'High' if risk_score > 7 else 'Medium' if risk_score > 4 else 'Low'
        summary_section += f"Overall Risk Score: {risk_score:.1f}/10.0 ({risk_level})\n"
        summary_section += f"Key Risks Identified: {len(document_data.get('key_risks', []))}\n"
        summary_section += f"Flagged Clauses: {len(document_data.get('flagged_clauses', []))}\n\n"
        
        report_sections.append(summary_section)
        
        # Risk Analysis
        risk_section = "RISK ANALYSIS\n" + "-" * 15 + "\n"
        
        # Risk categories
        risk_categories = document_data.get('risk_categories', {})
        if risk_categories:
            risk_section += "Risk Categories:\n"
            for category, score in risk_categories.items():
                level = 'High' if score > 7 else 'Medium' if score > 4 else 'Low'
                category_name = category.replace('_', ' ').title()
                risk_section += f"  • {category_name}: {score:.1f}/10.0 ({level})\n"
            risk_section += "\n"
        
        # Key risks
        key_risks = document_data.get('key_risks', [])
        if key_risks:
            risk_section += "Key Risks Identified:\n"
            for i, risk in enumerate(key_risks, 1):
                risk_section += f"  {i}. {risk}\n"
            risk_section += "\n"
        
        # Flagged clauses
        flagged_clauses = document_data.get('flagged_clauses', [])
        if flagged_clauses:
            risk_section += "Problematic Clauses:\n"
            for clause in flagged_clauses[:5]:  # Limit to top 5
                risk_section += f"  • {clause.get('issue_type', 'Issue').title()}: {clause.get('description', 'No description available.')}\n"
                risk_section += f"    Recommendation: {clause.get('recommendation', 'Review with legal counsel.')}\n\n"
        
        report_sections.append(risk_section)
        
        # Key Findings
        findings_section = "KEY FINDINGS\n" + "-" * 15 + "\n"
        
        # User obligations
        obligations = document_data.get('user_obligations', [])
        if obligations:
            findings_section += "Your Key Obligations:\n"
            for obligation in obligations[:7]:  # Limit to top 7
                if isinstance(obligation, dict):
                    obligation_text = obligation.get('obligation', str(obligation))
                else:
                    obligation_text = str(obligation)
                findings_section += f"  • {obligation_text}\n"
            findings_section += "\n"
        
        # User rights
        rights = document_data.get('user_rights', [])
        if rights:
            findings_section += "Your Rights:\n"
            for right in rights[:7]:  # Limit to top 7
                if isinstance(right, dict):
                    right_text = right.get('right', str(right))
                else:
                    right_text = str(right)
                findings_section += f"  • {right_text}\n"
            findings_section += "\n"
        
        # Financial implications
        financial = document_data.get('financial_implications', {})
        if financial:
            findings_section += "Financial Implications:\n"
            for key, value in financial.items():
                if isinstance(value, list) and value:
                    value_text = ', '.join(value[:3])  # Limit to first 3 items
                    if len(value) > 3:
                        value_text += f' (and {len(value) - 3} more)'
                else:
                    value_text = str(value) if value else 'None specified'
                
                key_formatted = key.replace('_', ' ').title()
                findings_section += f"  • {key_formatted}: {value_text}\n"
            findings_section += "\n"
        
        report_sections.append(findings_section)
        
        # Recommendations
        recommendations_section = "RECOMMENDATIONS\n" + "-" * 15 + "\n"
        
        recommendations = document_data.get('recommendations', [])
        if recommendations:
            recommendations_section += "Immediate Actions:\n"
            for i, recommendation in enumerate(recommendations, 1):
                recommendations_section += f"  {i}. {recommendation}\n"
            recommendations_section += "\n"
        
        # General guidance based on risk score
        recommendations_section += "General Guidance:\n"
        if risk_score > 7:
            guidance = [
                "Consider having this document reviewed by a qualified attorney before signing.",
                "Pay special attention to the high-risk clauses identified in this analysis.",
                "Negotiate terms that are unfavorable or unclear.",
                "Document any verbal agreements or modifications in writing."
            ]
        elif risk_score > 4:
            guidance = [
                "Review the flagged issues carefully and consider their impact.",
                "Clarify any unclear terms with the other party.",
                "Keep records of all communications regarding this agreement.",
                "Consider professional legal advice for complex clauses."
            ]
        else:
            guidance = [
                "This document appears to be relatively balanced.",
                "Still review all terms carefully before signing.",
                "Keep a copy of the signed agreement for your records.",
                "Monitor compliance with the agreed terms."
            ]
        
        for guidance_item in guidance:
            recommendations_section += f"  • {guidance_item}\n"
        
        report_sections.append(recommendations_section)
        
        # Legal disclaimer
        disclaimer_section = "\nLEGAL DISCLAIMER\n" + "-" * 20 + "\n"
        disclaimer_section += """This analysis is generated using artificial intelligence and is intended for informational 
and educational purposes only. It does not constitute legal advice, nor does it create 
an attorney-client relationship. For specific legal guidance, decisions, or actions 
related to this document, you should consult with a qualified attorney licensed to 
practice in your jurisdiction.

The creators and operators of this AI system make no warranties about the accuracy, 
completeness, or reliability of the analysis and disclaim any liability for decisions 
made based on this information.\n"""
        
        report_sections.append(disclaimer_section)
        
        # Combine all sections
        full_report = '\n'.join(report_sections)
        report_data_bytes = full_report.encode('utf-8')
        
        # Save to storage
        report_id = document_data.get('document_id', f'report_{uuid.uuid4().hex[:8]}')
        
        # For text reports, we'll save as .txt file
        try:
            download_url = await self.storage_manager.save_report_pdf(
                report_data_bytes, report_id, f"legal_analysis_{report_id}.txt"
            )
        except Exception as e:
            self.logger.error(f"Storage save failed: {e}")
            download_url = f"/downloads/legal_analysis_{report_id}.txt"
        
        # Create report metadata
        report_data = {
            'report_id': report_id,
            'title': f"Legal Document Analysis - {document_data.get('title', 'Document')}",
            'document_type': document_data.get('document_type', 'Unknown'),
            'risk_score': document_data.get('overall_risk_score', 0),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'language': language,
            'report_type': report_type,
            'page_count': len(full_report) // 2500,  # Estimate based on characters
            'file_size': len(report_data_bytes),
            'sections': [
                'Executive Summary',
                'Risk Analysis',
                'Key Findings',
                'Recommendations',
                'Legal Disclaimer'
            ],
            'charts_included': False,  # No charts in text format
            'format': 'TEXT'
        }
        
        self.logger.info(f"Text Report generated successfully: {report_id}")
        
        return {
            'report_data': report_data,
            'download_url': download_url,
            'pdf_size': len(report_data_bytes),
            'report_text': full_report  # Include the full text for immediate use
        }

    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for ReportLab"""
        
        if not REPORTLAB_AVAILABLE:
            return {}
        
        base_styles = getSampleStyleSheet()
        
        custom_styles = {
            'CustomTitle': ParagraphStyle(
                'CustomTitle',
                parent=base_styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=self.colors['primary'],
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ),
            'CustomHeading1': ParagraphStyle(
                'CustomHeading1',
                parent=base_styles['Heading1'],
                fontSize=18,
                spaceAfter=12,
                spaceBefore=20,
                textColor=self.colors['dark'],
                fontName='Helvetica-Bold'
            ),
            'CustomHeading2': ParagraphStyle(
                'CustomHeading2',
                parent=base_styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                spaceBefore=15,
                textColor=self.colors['dark'],
                fontName='Helvetica-Bold'
            ),
            'CustomBody': ParagraphStyle(
                'CustomBody',
                parent=base_styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName='Helvetica'
            ),
            'RiskHigh': ParagraphStyle(
                'RiskHigh',
                parent=base_styles['Normal'],
                fontSize=11,
                textColor=self.colors['danger'],
                fontName='Helvetica-Bold'
            ),
            'RiskMedium': ParagraphStyle(
                'RiskMedium',
                parent=base_styles['Normal'],
                fontSize=11,
                textColor=self.colors['warning'],
                fontName='Helvetica-Bold'
            ),
            'RiskLow': ParagraphStyle(
                'RiskLow',
                parent=base_styles['Normal'],
                fontSize=11,
                textColor=self.colors['secondary'],
                fontName='Helvetica-Bold'
            ),
            'Recommendation': ParagraphStyle(
                'Recommendation',
                parent=base_styles['Normal'],
                fontSize=11,
                leftIndent=20,
                bulletIndent=10,
                fontName='Helvetica'
            )
        }
        
        return custom_styles

    def _create_title_page(self, document_data: Dict[str, Any], report_type: str) -> List:
        """Create report title page (ReportLab version)"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        # Main title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("LegalMind AI", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Subtitle
        story.append(Paragraph(
            f"Legal Document Analysis Report",
            self.styles['CustomHeading1']
        ))
        story.append(Spacer(1, 40))
        
        # Document information table
        doc_info_data = [
            ['Document Title:', document_data.get('title', 'Legal Document')],
            ['Document Type:', document_data.get('document_type', 'Unknown').replace('_', ' ').title()],
            ['Analysis Date:', datetime.now().strftime('%B %d, %Y')],
            ['Report Type:', report_type.title()],
            ['Risk Score:', f"{document_data.get('overall_risk_score', 0):.1f}/10.0"],
            ['Pages Analyzed:', str(document_data.get('page_count', 'N/A'))]
        ]
        
        doc_info_table = Table(doc_info_data, colWidths=[2*inch, 3*inch])
        doc_info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(doc_info_table)
        story.append(Spacer(1, 60))
        
        # Disclaimer
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b><br/>
        This analysis is generated by AI and is intended for informational purposes only. 
        It does not constitute legal advice. For specific legal guidance, please consult 
        with a qualified attorney. The analysis is based on the document provided and 
        may not capture all nuances or implications.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['CustomBody']))
        
        return story

    def _create_executive_summary(self, document_data: Dict[str, Any]) -> List:
        """Create executive summary section (ReportLab version)"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        
        # Document summary
        summary = document_data.get('summary', 'No summary available.')
        story.append(Paragraph(summary, self.styles['CustomBody']))
        story.append(Spacer(1, 15))
        
        # Key metrics table
        risk_score = document_data.get('overall_risk_score', 0)
        risk_level = 'High' if risk_score > 7 else 'Medium' if risk_score > 4 else 'Low'
        
        metrics_data = [
            ['Metric', 'Value', 'Assessment'],
            ['Overall Risk Score', f'{risk_score:.1f}/10.0', risk_level],
            ['Key Risks Identified', str(len(document_data.get('key_risks', []))), 'Needs Review'],
            ['Flagged Clauses', str(len(document_data.get('flagged_clauses', []))), 'Attention Required'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
        ]))
        
        story.append(metrics_table)
        
        return story

    def _create_risk_analysis_section(self, document_data: Dict[str, Any], include_charts: bool) -> List:
        """Create risk analysis section (ReportLab version)"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        story.append(Paragraph("Risk Analysis", self.styles['CustomHeading1']))
        
        # Risk categories
        risk_categories = document_data.get('risk_categories', {})
        if risk_categories:
            story.append(Paragraph("Risk Categories", self.styles['CustomHeading2']))
            
            # Create risk categories table
            risk_data = [['Category', 'Score', 'Level']]
            
            for category, score in risk_categories.items():
                level = 'High' if score > 7 else 'Medium' if score > 4 else 'Low'
                category_name = category.replace('_', ' ').title()
                risk_data.append([category_name, f'{score:.1f}', level])
            
            risk_table = Table(risk_data, colWidths=[2.5*inch, 1*inch, 1*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ]))
            
            story.append(risk_table)
            story.append(Spacer(1, 15))
        
        # Key risks
        key_risks = document_data.get('key_risks', [])
        if key_risks:
            story.append(Paragraph("Key Risks Identified", self.styles['CustomHeading2']))
            
            for i, risk in enumerate(key_risks, 1):
                risk_text = f"{i}. {risk}"
                story.append(Paragraph(risk_text, self.styles['RiskHigh']))
            
            story.append(Spacer(1, 15))
        
        # Flagged clauses
        flagged_clauses = document_data.get('flagged_clauses', [])
        if flagged_clauses:
            story.append(Paragraph("Problematic Clauses", self.styles['CustomHeading2']))
            
            for clause in flagged_clauses[:5]:  # Limit to top 5
                clause_text = f"<b>{clause.get('issue_type', 'Issue').title()}:</b> {clause.get('description', 'No description available.')}"
                story.append(Paragraph(clause_text, self.styles['CustomBody']))
                
                rec_text = f"<i>Recommendation:</i> {clause.get('recommendation', 'Review with legal counsel.')}"
                story.append(Paragraph(rec_text, self.styles['Recommendation']))
                story.append(Spacer(1, 10))
        
        return story

    def _create_key_findings_section(self, document_data: Dict[str, Any]) -> List:
        """Create key findings section (ReportLab version)"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        story.append(Paragraph("Key Findings", self.styles['CustomHeading1']))
        
        # User obligations
        obligations = document_data.get('user_obligations', [])
        if obligations:
            story.append(Paragraph("Your Key Obligations", self.styles['CustomHeading2']))
            
            for obligation in obligations[:7]:  # Limit to top 7
                if isinstance(obligation, dict):
                    obligation_text = obligation.get('obligation', str(obligation))
                else:
                    obligation_text = str(obligation)
                
                story.append(Paragraph(f"• {obligation_text}", self.styles['CustomBody']))
            
            story.append(Spacer(1, 15))
        
        # User rights
        rights = document_data.get('user_rights', [])
        if rights:
            story.append(Paragraph("Your Rights", self.styles['CustomHeading2']))
            
            for right in rights[:7]:  # Limit to top 7
                if isinstance(right, dict):
                    right_text = right.get('right', str(right))
                else:
                    right_text = str(right)
                
                story.append(Paragraph(f"• {right_text}", self.styles['CustomBody']))
            
            story.append(Spacer(1, 15))
        
        # Financial implications
        financial = document_data.get('financial_implications', {})
        if financial:
            story.append(Paragraph("Financial Implications", self.styles['CustomHeading2']))
            
            # Create financial summary table
            financial_data = [['Type', 'Details']]
            
            for key, value in financial.items():
                if isinstance(value, list) and value:
                    value_text = ', '.join(value[:3])  # Limit to first 3 items
                    if len(value) > 3:
                        value_text += f' (and {len(value) - 3} more)'
                else:
                    value_text = str(value) if value else 'None specified'
                
                key_formatted = key.replace('_', ' ').title()
                financial_data.append([key_formatted, value_text])
            
            if len(financial_data) > 1:  # Only create table if there's data
                financial_table = Table(financial_data, colWidths=[2*inch, 3.5*inch])
                financial_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.colors['secondary']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), white),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
                ]))
                
                story.append(financial_table)
        
        return story

    def _create_recommendations_section(self, document_data: Dict[str, Any]) -> List:
        """Create recommendations section (ReportLab version)"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        story.append(Paragraph("Recommendations", self.styles['CustomHeading1']))
        
        recommendations = document_data.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("Immediate Actions", self.styles['CustomHeading2']))
            
            for i, recommendation in enumerate(recommendations, 1):
                rec_text = f"{i}. {recommendation}"
                story.append(Paragraph(rec_text, self.styles['CustomBody']))
            
            story.append(Spacer(1, 15))
        
        # General recommendations based on risk score
        risk_score = document_data.get('overall_risk_score', 0)
        
        story.append(Paragraph("General Guidance", self.styles['CustomHeading2']))
        
        if risk_score > 7:
            guidance = [
                "Consider having this document reviewed by a qualified attorney before signing.",
                "Pay special attention to the high-risk clauses identified in this analysis.",
                "Negotiate terms that are unfavorable or unclear.",
                "Document any verbal agreements or modifications in writing."
            ]
        elif risk_score > 4:
            guidance = [
                "Review the flagged issues carefully and consider their impact.",
                "Clarify any unclear terms with the other party.",
                "Keep records of all communications regarding this agreement.",
                "Consider professional legal advice for complex clauses."
            ]
        else:
            guidance = [
                "This document appears to be relatively balanced.",
                "Still review all terms carefully before signing.",
                "Keep a copy of the signed agreement for your records.",
                "Monitor compliance with the agreed terms."
            ]
        
        for guidance_item in guidance:
            story.append(Paragraph(f"• {guidance_item}", self.styles['CustomBody']))
        
        return story

    def _create_detailed_analysis_section(self, document_data: Dict[str, Any]) -> List:
        """Create detailed analysis section (for comprehensive reports)"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Detailed Analysis", self.styles['CustomHeading1']))
        
        # Industry comparison
        fairness_score = document_data.get('fairness_score', 5.0)
        story.append(Paragraph("Industry Comparison", self.styles['CustomHeading2']))
        
        fairness_text = f"""
        This document has a fairness score of {fairness_score:.1f}/10.0 compared to industry standards. 
        """
        
        if fairness_score > 7:
            fairness_text += "This is considered favorable to you as the document holder."
        elif fairness_score > 4:
            fairness_text += "This is considered reasonably balanced between parties."
        else:
            fairness_text += "This document may be more favorable to the other party."
        
        story.append(Paragraph(fairness_text, self.styles['CustomBody']))
        story.append(Spacer(1, 15))
        
        # Document structure analysis
        story.append(Paragraph("Document Structure", self.styles['CustomHeading2']))
        
        structure_data = [
            ['Element', 'Status', 'Notes'],
            ['Clear Terms', '✓' if document_data.get('overall_risk_score', 5) < 6 else '⚠', 'Most terms are clearly defined'],
            ['Fair Termination', '✓' if 'termination' not in str(document_data.get('key_risks', [])).lower() else '⚠', 'Termination clauses reviewed'],
            ['Reasonable Penalties', '✓' if document_data.get('overall_risk_score', 5) < 7 else '⚠', 'Penalty terms assessed'],
            ['Balanced Rights', '✓' if fairness_score > 5 else '⚠', 'Rights distribution analyzed'],
        ]
        
        structure_table = Table(structure_data, colWidths=[2*inch, 1*inch, 2.5*inch])
        structure_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['dark']),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
        ]))
        
        story.append(structure_table)
        
        return story

    def _create_appendix_section(self, document_data: Dict[str, Any]) -> List:
        """Create appendix section"""
        
        if not REPORTLAB_AVAILABLE:
            return []
        
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Appendix", self.styles['CustomHeading1']))
        
        # Analysis metadata
        story.append(Paragraph("Analysis Details", self.styles['CustomHeading2']))
        
        metadata = document_data.get('metadata', {})
        
        metadata_data = [
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M UTC')],
            ['AI Model Version', metadata.get('model_version', 'Gemini 2.5 Pro')],
            ['Processing Time', str(document_data.get('processing_time', 'Unknown'))],
            ['Document Words', str(metadata.get('word_count', 'Unknown'))],
            ['Confidence Level', 'High (AI Analysis)'],
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Legal disclaimer
        story.append(Paragraph("Legal Disclaimer", self.styles['CustomHeading2']))
        
        disclaimer = """
        This analysis is generated using artificial intelligence and is intended for informational 
        and educational purposes only. It does not constitute legal advice, nor does it create 
        an attorney-client relationship. The analysis is based solely on the document provided 
        and may not account for all legal nuances, jurisdictional variations, or specific 
        circumstances that may affect the interpretation or enforceability of the document.

        For specific legal guidance, decisions, or actions related to this document, you should 
        consult with a qualified attorney licensed to practice in your jurisdiction. The AI 
        analysis should be used as a starting point for understanding potential issues, not as 
        a substitute for professional legal counsel.

        The creators and operators of this AI system make no warranties about the accuracy, 
        completeness, or reliability of the analysis and disclaim any liability for decisions 
        made based on this information.
        """
        
        story.append(Paragraph(disclaimer, self.styles['CustomBody']))
        
        return story

    async def generate_risk_summary_chart(self, risk_categories: Dict[str, float]) -> bytes:
        """Generate risk summary chart as image"""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - cannot generate charts")
            return b''
        
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            categories = list(risk_categories.keys())
            scores = list(risk_categories.values())
            
            # Create color map based on risk levels
            colors = []
            for score in scores:
                if score > 7:
                    colors.append('#ea4335')  # Red for high risk
                elif score > 4:
                    colors.append('#fbbc04')  # Yellow for medium risk
                else:
                    colors.append('#34a853')  # Green for low risk
            
            # Create bar chart
            bars = ax.bar(categories, scores, color=colors)
            
            # Customize chart
            ax.set_title('Risk Assessment by Category', fontsize=16, fontweight='bold')
            ax.set_ylabel('Risk Score (0-10)', fontsize=12)
            ax.set_ylim(0, 10)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add legend
            import matplotlib.patches as mpatches
            high_patch = mpatches.Patch(color='#ea4335', label='High Risk (7-10)')
            medium_patch = mpatches.Patch(color='#fbbc04', label='Medium Risk (4-7)')
            low_patch = mpatches.Patch(color='#34a853', label='Low Risk (0-4)')
            ax.legend(handles=[high_patch, medium_patch, low_patch], loc='upper right')
            
            plt.tight_layout()
            
            # Convert to bytes
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            chart_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error generating risk chart: {str(e)}")
            return b''  # Return empty bytes if chart generation fails

    def get_report_templates(self) -> List[Dict[str, Any]]:
        """Get available report templates"""
        
        templates = [
            {
                'id': 'brief',
                'name': 'Brief Analysis',
                'description': 'Quick overview with key risks and recommendations',
                'sections': ['Executive Summary', 'Key Risks', 'Recommendations'],
                'estimated_pages': 3,
                'format': 'PDF' if REPORTLAB_AVAILABLE else 'TEXT'
            },
            {
                'id': 'comprehensive',
                'name': 'Comprehensive Analysis',
                'description': 'Detailed analysis with all sections and appendices',
                'sections': ['Executive Summary', 'Risk Analysis', 'Key Findings', 'Recommendations', 'Detailed Analysis', 'Appendix'],
                'estimated_pages': 8,
                'format': 'PDF' if REPORTLAB_AVAILABLE else 'TEXT'
            },
            {
                'id': 'risk_focused',
                'name': 'Risk-Focused Report',
                'description': 'Concentrated on risk identification and mitigation',
                'sections': ['Risk Analysis', 'Flagged Clauses', 'Risk Mitigation', 'Recommendations'],
                'estimated_pages': 5,
                'format': 'PDF' if REPORTLAB_AVAILABLE else 'TEXT'
            }
        ]
        
        return templates

    def get_generator_capabilities(self) -> Dict[str, Any]:
        """Get information about report generator capabilities"""
        
        return {
            'pdf_generation': REPORTLAB_AVAILABLE,
            'chart_generation': MATPLOTLIB_AVAILABLE,
            'text_fallback': True,
            'storage_integration': STORAGE_AVAILABLE,
            'supported_formats': ['PDF', 'TEXT'] if REPORTLAB_AVAILABLE else ['TEXT'],
            'report_types': ['brief', 'comprehensive', 'risk_focused'],
            'languages': ['en'],  # Expandable
            'features': {
                'custom_styles': REPORTLAB_AVAILABLE,
                'tables': REPORTLAB_AVAILABLE,
                'charts': MATPLOTLIB_AVAILABLE,
                'professional_layout': REPORTLAB_AVAILABLE,
                'legal_disclaimers': True,
                'risk_visualization': MATPLOTLIB_AVAILABLE
            }
        }


# Utility functions
async def generate_simple_report(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple report generation for basic use cases"""
    
    generator = ReportGenerator()
    
    try:
        return await generator.generate_comprehensive_report(
            document_data=document_data,
            report_type="brief",
            language="en",
            include_charts=True
        )
    except Exception as e:
        logger.error(f"Simple report generation failed: {e}")
        return {
            'report_data': {
                'report_id': 'error',
                'title': 'Report Generation Failed',
                'error': str(e)
            },
            'download_url': None,
            'pdf_size': 0
        }

def create_risk_summary_text(risk_categories: Dict[str, float]) -> str:
    """Create text-based risk summary when charts aren't available"""
    
    if not risk_categories:
        return "No risk data available."
    
    summary_lines = ["RISK SUMMARY:", "=" * 15]
    
    for category, score in risk_categories.items():
        level = 'HIGH' if score > 7 else 'MEDIUM' if score > 4 else 'LOW'
        category_name = category.replace('_', ' ').title()
        
        # Create simple text bar
        bar_length = int(score)
        bar = '█' * bar_length + '░' * (10 - bar_length)
        
        summary_lines.append(f"{category_name:20} {score:4.1f}/10 [{bar}] {level}")
    
    return '\n'.join(summary_lines)
