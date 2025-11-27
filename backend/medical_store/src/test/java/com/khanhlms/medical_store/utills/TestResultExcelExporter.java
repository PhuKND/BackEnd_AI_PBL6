package com.khanhlms.medical_store.utills;

import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.util.List;
import java.util.Map;

@Slf4j
public class TestResultExcelExporter {

    private static final String MAIN_SHEET = "Results";
    private static final String SUMMARY_SHEET = "Summary";

    // Ghi tiếp dữ liệu vào 1 sheet duy nhất
    public static void appendResults(
            String fileName,
            String moduleName,
            List<Map<String, String>> results,
            List<String> headers
    ) {
        Workbook workbook;
        File file = new File(fileName);

        try {

            // 1. Mở file nếu đã có
            if (file.exists()) {
                FileInputStream fis = new FileInputStream(file);
                workbook = new XSSFWorkbook(fis);
                fis.close();
            } else {
                workbook = new XSSFWorkbook();
            }

            // 2. Tạo sheet Results nếu chưa có
            Sheet sheet = workbook.getSheet(MAIN_SHEET);
            if (sheet == null) {
                sheet = workbook.createSheet(MAIN_SHEET);

                // tạo header (thêm cột Module)
                Row headerRow = sheet.createRow(0);
                for (int i = 0; i < headers.size(); i++) {
                    headerRow.createCell(i).setCellValue(headers.get(i));
                }
                headerRow.createCell(headers.size()).setCellValue("Module");
            }

            // 3. Ghi tiếp dòng mới
            int lastRow = sheet.getLastRowNum() + 1;

            for (Map<String, String> record : results) {
                Row row = sheet.createRow(lastRow++);

                for (int i = 0; i < headers.size(); i++) {
                    row.createCell(i).setCellValue(record.get(headers.get(i)));
                }

                // thêm tên module
                row.createCell(headers.size()).setCellValue(moduleName);
            }

            // Auto size
            for (int i = 0; i <= headers.size(); i++) {
                sheet.autoSizeColumn(i);
            }

            // 4. Ghi ra file
            FileOutputStream fos = new FileOutputStream(fileName);
            workbook.write(fos);
            fos.close();
            workbook.close();

            log.info("✔ Appended results for module: " + moduleName);

        } catch (Exception e) {
            log.error("❌ Error appending results: " + e.getMessage());
        }
    }


    // Tạo sheet thống kê
    public static void exportSummary(String fileName) {
        try {
            FileInputStream fis = new FileInputStream(fileName);
            Workbook workbook = new XSSFWorkbook(fis);
            fis.close();

            Sheet resultSheet = workbook.getSheet(MAIN_SHEET);
            if (resultSheet == null) return;

            // xóa summary cũ nếu có
            int idx = workbook.getSheetIndex(SUMMARY_SHEET);
            if (idx != -1) workbook.removeSheetAt(idx);

            Sheet summary = workbook.createSheet(SUMMARY_SHEET);

            int total = resultSheet.getLastRowNum();
            int passed = 0;

            for (int i = 1; i <= resultSheet.getLastRowNum(); i++) {
                Row r = resultSheet.getRow(i);
                if (r == null) continue;

                String status = r.getCell(2).getStringCellValue();
                if ("PASSED".equalsIgnoreCase(status)) passed++;
            }

            int failed = total - passed;

            summary.createRow(0).createCell(0).setCellValue("Total Tests");
            summary.getRow(0).createCell(1).setCellValue(total);

            summary.createRow(1).createCell(0).setCellValue("Passed");
            summary.getRow(1).createCell(1).setCellValue(passed);

            summary.createRow(2).createCell(0).setCellValue("Failed");
            summary.getRow(2).createCell(1).setCellValue(failed);

            summary.createRow(3).createCell(0).setCellValue("Pass Rate (%)");
            summary.getRow(3).createCell(1).setCellValue(total == 0 ? 0 : passed * 100.0 / total);

            summary.createRow(4).createCell(0).setCellValue("Fail Rate (%)");
            summary.getRow(4).createCell(1).setCellValue(total == 0 ? 0 : failed * 100.0 / total);

            FileOutputStream fos = new FileOutputStream(fileName);
            workbook.write(fos);
            fos.close();
            workbook.close();

            log.info("📊 Summary created!");

        } catch (Exception e) {
            log.error("❌ Unable to create summary: " + e.getMessage());
        }
    }
}
