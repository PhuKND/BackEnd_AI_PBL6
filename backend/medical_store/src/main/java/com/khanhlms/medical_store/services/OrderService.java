package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.configuration.VnPayConfig;
import com.khanhlms.medical_store.dtos.orders.request.CreateOrderRequest;
import com.khanhlms.medical_store.dtos.orders.request.ItemOrder;
import com.khanhlms.medical_store.dtos.orders.response.CreateOrderResponse;
import com.khanhlms.medical_store.entities.*;
import com.khanhlms.medical_store.enums.OrderStatus;
import com.khanhlms.medical_store.enums.PaymentMethod;
import com.khanhlms.medical_store.enums.PaymentStatus;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.repositories.OrderRepository;
import com.khanhlms.medical_store.repositories.ProductRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import com.khanhlms.medical_store.utills.RequestHttpUitlls;
import jakarta.servlet.http.HttpServletRequest;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

import java.util.LinkedList;
import java.util.List;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class OrderService {
    final OrderRepository orderRepository;
    final UserRepository userRepository;
    final ProductRepository productRepository;
    final VnPayService vnPayService;

    public CreateOrderResponse createOrder(HttpServletRequest httpServletRequest, String username, CreateOrderRequest request) {
        List<ItemOrder> itemOrders = request.getItemOrders();
        List<OrderItemEntity> orderItems = new LinkedList<>();
        Double totalAmount = 0.0 ;
        UserEntity user = userRepository.findByUsername(username).get();
        String status = request.getPaymentType().equals(PaymentMethod.COD.toString()) == true
                ? OrderStatus.PENDING.toString()
                : OrderStatus.CONFIRMED.toString();
        for (ItemOrder itemOrder : itemOrders) {
            ProductsEntity product = productRepository.findById(itemOrder.getProductId())
                    .orElseThrow(() -> new AppException(ErrorCode.PRODUCT_NOT_FOUND));
            Double price = product.getOriginPrice();
            Double percent = product.getDiscount() == null || product.getDiscount().getPercent() == null
                    ? 0
                    : product.getDiscount().getPercent();
            Double discountPrice = price * (100 - percent) / 100;
            Integer quantity = itemOrder.getQuantity();
            Double totalPrice = discountPrice * quantity;
            totalAmount = totalAmount + totalPrice;
            OrderItemEntity orderItem = OrderItemEntity.builder()
                    .product(product)
                    .price(price)
                    .discountPrice(discountPrice)
                    .quantity(quantity)
                    .total(totalPrice)
                    .product(product)
                    .build();
            orderItems.add(orderItem);
        }
        PaymentEntity paymentEntity = PaymentEntity.builder()
                .paymentNote(request.getPaymentNote())
                .paymentMethod(PaymentMethod.valueOf(request.getPaymentType()).toString())
                .status(PaymentStatus.PENDING.toString())
                .build();

        OrderEntity orderEntity = OrderEntity.builder()
                .totalAmount(totalAmount)
                .orderItems(orderItems)
                .payment(paymentEntity)
                .note(request.getNote())
                .address(request.getAddress())
                .city(request.getCity())
                .ward(request.getWard())
                .phoneNumber(request.getPhoneNumber())
                .status(status)
                .user(user)
                .build();

        // 🔥 Gán quan hệ 2 chiều
        paymentEntity.setOrder(orderEntity);

        for (OrderItemEntity item : orderItems) {
            item.setOrder(orderEntity);
        }
        // 🔥 Persist vào DB
        orderEntity = orderRepository.save(orderEntity);

        // 🔥 Xử lý thanh toán VNPay nếu chọn online
        String redirectUrl = null;
        if (paymentEntity.getPaymentMethod().equals(PaymentMethod.VNPAY.toString())) {
            redirectUrl = vnPayService.createPaymentUrl(
                    orderEntity.getTotalAmount().longValue(),
                    orderEntity.getId(),
                    orderEntity.getPayment().getId(),
                    RequestHttpUitlls.getClientIp(httpServletRequest)
            );
        }
        return CreateOrderResponse.builder()
                .orderId(orderEntity.getId())
                .paymentMethod(orderEntity.getPayment().getPaymentMethod())
                .totalAmount(orderEntity.getTotalAmount())
                .status(orderEntity.getStatus())
                .redirectUrl(redirectUrl)
                .build();

    }

}
