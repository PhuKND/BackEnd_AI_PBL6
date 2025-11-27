package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.utills.TestResultExcelExporter;
import org.junit.jupiter.api.Test;

public class SummaryTest {

    @Test
    void summary() {
        TestResultExcelExporter.exportSummary("test-result.xlsx");
    }

}
