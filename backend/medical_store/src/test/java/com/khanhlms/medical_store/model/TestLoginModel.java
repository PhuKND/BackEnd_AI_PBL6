package com.khanhlms.medical_store.model;

import lombok.AccessLevel;
import lombok.Data;
import lombok.experimental.FieldDefaults;

@Data
@FieldDefaults(level = AccessLevel.PRIVATE)
public class TestLoginModel {
    String testCaseID;
    String testCaseDescription;
    String username;
    String password;
    Integer expectedStatus;
    String expectedMessage;
}
