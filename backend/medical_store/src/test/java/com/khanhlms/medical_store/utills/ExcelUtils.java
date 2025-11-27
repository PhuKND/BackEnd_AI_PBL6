package com.khanhlms.medical_store.utills;

import com.khanhlms.medical_store.model.CategotiesModel;
import com.khanhlms.medical_store.model.RegisterModel;
import com.khanhlms.medical_store.model.TestLoginModel;
import com.khanhlms.medical_store.model.TestManufacturerModel;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ExcelUtils {

    public static List<TestLoginModel> readLoginTestData(String excelFilePath) {
        List<TestLoginModel> testCases = new ArrayList<>();

        try (InputStream inputStream = ExcelUtils.class.getResourceAsStream(excelFilePath);
             Workbook workbook = new XSSFWorkbook(inputStream)) {

            Sheet sheet = workbook.getSheetAt(0);

            // Bỏ header (bắt đầu từ row index 1)
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);

                TestLoginModel testCase = new TestLoginModel();
                testCase.setTestCaseID(getStringCell(row.getCell(0)));
                testCase.setTestCaseDescription(getStringCell(row.getCell(1)));
                testCase.setUsername(getStringCell(row.getCell(2)));
                testCase.setPassword(getStringCell(row.getCell(3)));
                testCase.setExpectedStatus(getIntCell(row.getCell(4)));
                testCase.setExpectedMessage(getStringCell(row.getCell(5)).trim()); // ✅ TRIM ở đây

                testCases.add(testCase);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return testCases;
    }

    private static String getStringCell(Cell cell) {
        if (cell == null) return "";
        if (cell.getCellType() == CellType.NUMERIC) {
            return String.valueOf((int) cell.getNumericCellValue()).trim();
        }
        return cell.getStringCellValue().trim(); // ✅ TRIM thêm tại đây luôn
    }

    private static Integer getIntCell(Cell cell) {
        return (cell == null) ? null : (int) cell.getNumericCellValue();
    }

    public static List<TestManufacturerModel> readManufacturerTestData(String excelFilePath) {
        List<TestManufacturerModel> testCases = new ArrayList<>();

        try (InputStream inputStream = ExcelUtils.class.getResourceAsStream(excelFilePath);
             Workbook workbook = new XSSFWorkbook(inputStream)) {

            Sheet sheet = workbook.getSheetAt(1); // ✅ sheet thứ 2
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {

                Row row = sheet.getRow(i);
                TestManufacturerModel tc = new TestManufacturerModel();

                tc.setTestCaseID(getStringCell(row.getCell(0)));
                tc.setTestCaseDescription(getStringCell(row.getCell(1)));
                tc.setName(getStringCell(row.getCell(2)));
                tc.setToken(getStringCell(row.getCell(3)));
                tc.setExpectedStatus(getIntCell(row.getCell(4)));
                tc.setExpectedMessage(getStringCell(row.getCell(5)));

                testCases.add(tc);
                System.out.println(tc.toString());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return testCases;
    }
    public static List<RegisterModel> readRegisterTestData(String excelFilePath) {
        List<RegisterModel> testCases = new ArrayList<>();

        try (InputStream inputStream = ExcelUtils.class.getResourceAsStream(excelFilePath);
             Workbook workbook = new XSSFWorkbook(inputStream)) {

            Sheet sheet = workbook.getSheetAt(2); // Sheet register

            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);

                RegisterModel tc = new RegisterModel();
                tc.setTestCaseID(getStringCell(row.getCell(0)));             // ✅ Test Case ID
                tc.setTestCaseDescription(getStringCell(row.getCell(1)));    // ✅ Description
                tc.setUsername(getStringCell(row.getCell(2)));               // Username
                tc.setPassword(getStringCell(row.getCell(3)));               // Password
                tc.setRepeat(getStringCell(row.getCell(4)));                 // Repeat
                tc.setExpectedStatus(getIntCell(row.getCell(5)));           // Expected status
                tc.setExpectedMessage(getStringCell(row.getCell(6)).trim()); // Expected message

                testCases.add(tc);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return testCases;
    }
    public static List<CategotiesModel> readCategoryTestData(String excelFilePath) {
        List<CategotiesModel> testCases = new ArrayList<>();

        try (InputStream inputStream = ExcelUtils.class.getResourceAsStream(excelFilePath);
             Workbook workbook = new XSSFWorkbook(inputStream)) {

            Sheet sheet = workbook.getSheetAt(3); // ✅ SHEET CATEGORY = sheet thứ 3

            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                if (row == null) continue;

                CategotiesModel tc = new CategotiesModel();

                tc.setTestCaseID(getStringCell(row.getCell(0)));            // A
                tc.setTestCaseDescription(getStringCell(row.getCell(1)));   // B
                tc.setToken(getStringCell(row.getCell(2)));                 // C
                tc.setName(getStringCell(row.getCell(3)));                  // D
                tc.setExpectedStatus(getIntCell(row.getCell(4)));           // E
                tc.setExpectedMessage(getStringCell(row.getCell(5)).trim()); // F

                testCases.add(tc);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return testCases;
    }


}
