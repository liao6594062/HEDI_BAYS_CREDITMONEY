libname ww '\\file.tounawang.net\������ݹ���\llll\D_HE\1_����ί������ɸѡ\data';
libname b  '\\file.tounawang.net\����ֻ��\1. ����\3. sas���ݼ�';
libname c  '\\file.tounawang.net\����ֻ��\1. ����\3. sas���ݼ�\9.stg��';

%let user=hedi;
%let password=9ku3y3b9;
/*��Ҫ�޸�datadate����ϵͳ����*/
%let datadate=20190829;

/*�������ڱ���������*/
proc sql;
create table ww.Tbdsche_over as
select * from c.Tbdsche_over&datadate.
where currentduedays>=1;
quit;
PROC EXPORT DATA= ww.Tbdsche_over
            OUTFILE= "\\file.tounawang.net\������ݹ���\llll\D_HE\1_����ί������ɸѡ\data\Tbdsche_over&datadate..csv" 
            DBMS=CSV REPLACE;
     PUTNAMES=YES;
RUN;



