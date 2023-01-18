CREATE DATABASE IF NOT EXISTS `project_db`; -- choose database you want to create

USE `project_db`; -- select database you want to use

-- create item table
CREATE TABLE IF NOT EXISTS `item` (
	`item_id`	INT(10)	NOT NULL PRIMARY KEY,
	`category`	VARCHAR(100)	NULL,
	`rating`	VARCHAR(100)	NULL,
	`review`	VARCHAR(255)	NULL,
	`price`	VARCHAR(100)	NULL,
	`title`	VARCHAR(100)	NULL,
	`seller`	VARCHAR(100)	NULL,
	`discount_rate`	VARCHAR(100)	NULL,
	`image`	VARCHAR(255)	NULL,
	`available_product`	VARCHAR(100)	NULL,
	`predict_price`	VARCHAR(100)	NULL
);

-- create house table
CREATE TABLE IF NOT EXISTS `house` (
	`house_id`	INT(10)	NOT NULL PRIMARY KEY,
	`space`	VARCHAR(100)	NULL,
	`size`	VARCHAR(100)	NULL,
	`work`	VARCHAR(100)	NULL,
	`category`	VARCHAR(100)	NULL,
	`family`	VARCHAR(100)	NULL,
	`region`	VARCHAR(100)	NULL,
	`style`	VARCHAR(100)	NULL,
	`duration`	VARCHAR(100)	NULL,
	`budget`	VARCHAR(100)	NULL,
	`detail`	VARCHAR(255)	NULL,
	`prefer`	INT(10)	NULL,
	`scrab`	INT(10)	NULL,
	`comment`	INT(10)	NULL,
	`views`	INT(10)	NULL,
	`card_space` VARCHAR(100) 	NULL,
	`card_url`	VARCHAR(255)	NULL
);

-- create house_item interaction table
CREATE TABLE IF NOT EXISTS `house_item`(
	`house_id` INT(10) NOT NULL PRIMARY KEY,
	`item_id` INT(10) NOT NULL
);

-- create member table
CREATE TABLE IF NOT EXISTS `member` (
	`member_email`	VARCHAR(255)	NOT NULL PRIMARY KEY,
	`house_id`	INT(10)	NULL
);