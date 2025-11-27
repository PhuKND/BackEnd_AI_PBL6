package com.khanhlms.medical_store.enums;

public enum OrderStatus {
    PENDING,        // chờ xác nhận
    CONFIRMED,      // đã xác nhận
    SHIPPING,       // đang giao
    COMPLETED,      // giao thành công
    CANCELLED       // bị hủy
}
