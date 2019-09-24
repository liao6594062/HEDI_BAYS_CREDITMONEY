libname ww '\\file.tounawang.net\风控数据共享\llll\D_HE\1_车贷委外名单筛选\data';
libname b  '\\file.tounawang.net\公共只读\1. 车贷\3. sas数据集';
libname c  '\\file.tounawang.net\公共只读\1. 车贷\3. sas数据集\9.stg版';

%let user=hedi;
%let password=9ku3y3b9;
/*需要修改datadate保持系统最新*/
%let datadate=20190829;

/*进度逾期表逾期数据*/
proc sql;
create table ww.Tbdsche_over as
select * from c.Tbdsche_over&datadate.
where currentduedays>=1;
quit;
PROC EXPORT DATA= ww.Tbdsche_over
            OUTFILE= "\\file.tounawang.net\风控数据共享\llll\D_HE\1_车贷委外名单筛选\data\Tbdsche_over&datadate..csv" 
            DBMS=CSV REPLACE;
     PUTNAMES=YES;
RUN;



